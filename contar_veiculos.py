
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from math import dist
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2


PASTA = Path(__file__).resolve().parent
VIDEO = PASTA / "trafego_dia_noite.mp4"
MODELO = PASTA / "yolov5n.onnx"
CORTE_NOITE = 180  
LINHA_DIA = 300
FAIXAS_DIA = ((45, 284), (321, 469))
X_NOITE = 395
FAIXAS_NOITE = ((165, 311), (76, 125))
CLASSES = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}
TAMANHO = 640
AMARELO = (45, 210, 255)
VERDE = (176, 240, 83)


def detectar_veiculos(quadro, rede):
    
    altura, largura = quadro.shape[:2]
    escala_x, escala_y = largura / TAMANHO, altura / TAMANHO
    blob = cv2.dnn.blobFromImage(quadro, 1 / 255, (TAMANHO, TAMANHO), swapRB=True)
    rede.setInput(blob)
    saida = rede.forward(rede.getUnconnectedOutLayersNames())[0][0]
    caixas, tipos, confiancas = [], [], []
    for linha in saida[saida[:, 4] >= 0.45]:
        probabilidades = linha[5:]
        classe = int(probabilidades.argmax())
        if classe not in CLASSES or probabilidades[classe] <= 0.20:
            continue
        cx, cy, w, h = map(float, linha[:4])
        caixas.append(((cx - w / 2) * escala_x, (cy - h / 2) * escala_y,
                       w * escala_x, h * escala_y))
        tipos.append(CLASSES[classe])
        confiancas.append(float(linha[4]))

    indices = cv2.dnn.NMSBoxes(caixas, confiancas, 0.45, 0.45)
    return [(tuple(map(round, caixas[i])), tipos[i], confiancas[i]) for i in indices]


@dataclass(eq=False)
class Rastro:
    ponto: tuple
    quadro: int
    tipo: str
    confianca: float
    contado: bool = False


def contar_dia(quadro, indice, detector, rastros):
    rastros[:] = [r for r in rastros if indice - r.quadro <= 7]
    disponiveis = rastros.copy()
    deteccoes = detectar_veiculos(quadro, detector)
    passagens = []
    for (x, y, w, h), tipo, confianca in deteccoes:
        ponto = (x + w / 2, y + h)  
        candidatos = []
        for rastro in disponiveis:
            distancia = dist(ponto, rastro.ponto)
            limite = 14 + 5 * (indice - rastro.quadro)
            if distancia < limite:
                candidatos.append((distancia, rastro))
        if not candidatos:
            rastros.append(Rastro(ponto, indice, tipo, confianca))
            continue

        _, rastro = min(candidatos, key=lambda par: par[0])
        disponiveis.remove(rastro)
        if confianca > rastro.confianca:
            rastro.tipo, rastro.confianca = tipo, confianca

        x_anterior, y_anterior = rastro.ponto
        x_atual, y_atual = ponto
        estava_acima = y_anterior < LINHA_DIA
        cruzou = estava_acima != (y_atual < LINHA_DIA)
        if cruzou and not rastro.contado:
            faixa = 0 if estava_acima else 1
            esquerda, direita = FAIXAS_DIA[faixa]
            # Ve quando o  quadrado encosto na linha
            proporcao = (LINHA_DIA - y_anterior) / (y_atual - y_anterior)
            x_linha = x_anterior + proporcao * (x_atual - x_anterior)
            if esquerda <= x_linha <= direita:
                passagens.append((faixa, rastro.tipo))
                rastro.contado = True
        rastro.ponto, rastro.quadro = ponto, indice
    return passagens, deteccoes


def contar_noite(quadro, indice, ultimo_sinal):
    # Mascara hsv dos farois
    hsv = cv2.cvtColor(quadro, cv2.COLOR_BGR2HSV)
    coluna = hsv[:, X_NOITE - 4:X_NOITE + 5]
    (topo_perto, base_perto), (topo_longe, base_longe) = FAIXAS_NOITE
    perto = coluna[topo_perto:base_perto]
    longe = coluna[topo_longe:base_longe]
    farois = cv2.inRange(perto, (0, 0, 215), (179, 50, 255))
    vermelho_baixo = cv2.inRange(longe, (0, 40, 175), (24, 255, 255))
    vermelho_alto = cv2.inRange(longe, (166, 40, 175), (179, 255, 255))
    lanternas = cv2.bitwise_or(vermelho_baixo, vermelho_alto)
    sinais = (cv2.countNonZero(farois) >= 30, cv2.countNonZero(lanternas) >= 7)
    passagens = []
    for faixa, ativo in enumerate(sinais):
        if not ativo:
            continue
        if indice - ultimo_sinal[faixa] > 8:
            passagens.append((faixa, "veiculo"))
        ultimo_sinal[faixa] = indice
    return passagens


def anotar(quadro, periodo, totais, deteccoes, passagens):
    cor = VERDE if passagens else AMARELO
    if periodo == "dia":
        for x0, x1 in FAIXAS_DIA:
            cv2.line(quadro, (x0, LINHA_DIA), (x1, LINHA_DIA), cor, 2)
        for (x, y, w, h), tipo, _ in deteccoes:
            if 210 < y + h < 355:
                cv2.rectangle(quadro, (x, y), (x+w, y+h), (165, 216, 194), 1)
                cv2.putText(quadro, tipo, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (165, 216, 194), 1, cv2.LINE_AA)
    else:
        for topo, base in FAIXAS_NOITE:
            cv2.line(quadro, (X_NOITE, topo), (X_NOITE, base), cor, 2)
    cv2.rectangle(quadro, (0, 0), (639, 35), (21, 35, 42), -1)
    texto = f"{periodo.upper()}   |   Dia: {totais['dia']}   Noite: {totais['noite']}   Total: {sum(totais.values())}"
    cv2.putText(quadro, texto, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .64, (241, 246, 242), 1, cv2.LINE_AA)


def analisar(caminho, anotado):
    cv2.setNumThreads(3)
    detector = cv2.dnn.readNet(str(MODELO))
    cap = cv2.VideoCapture(str(caminho))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise ValueError(f"Não foi possível abrir {caminho}.")
    writer = cv2.VideoWriter(str(anotado), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Não foi possível gravar {anotado}.")

    eventos, rastros, fotos = [], [], {}
    totais = Counter()
    ultimo_sinal = [-100, -100]
    indice = 0
    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            if indice < CORTE_NOITE:
                periodo = "dia"
                passagens, deteccoes = contar_dia(quadro, indice, detector, rastros)
            else:
                periodo = "noite"
                passagens = contar_noite(quadro, indice, ultimo_sinal)
                deteccoes = []
            totais[periodo] += len(passagens)
            for faixa, tipo in passagens:
                eventos.append({"periodo": periodo, "quadro": indice + 1,
                                "tempo_s": round(indice / fps, 3),
                                "sentido": ("aproximacao", "afastamento")[faixa], "tipo": tipo})
            anotar(quadro, periodo, totais, deteccoes, passagens)
            writer.write(quadro)
            if passagens:
                fotos[periodo] = quadro.copy()
            indice += 1
    finally:
        cap.release()
        writer.release()
    return eventos, totais, fotos


def reproduzir(caminho):
    cap = cv2.VideoCapture(str(caminho))
    nome = "Contagem de veiculos - Esc para fechar"
    cv2.namedWindow(nome)
    espera = max(1, round(1000 / cap.get(cv2.CAP_PROP_FPS)))
    ultimo = None
    try:
        while True:
            ok, quadro = cap.read()
            if ok:
                ultimo = quadro
            if ultimo is None:
                break
            cv2.imshow(nome, ultimo)
            if cv2.waitKey(espera) & 0xFF in (27, ord("q")):
                break
            if cv2.getWindowProperty(nome, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def salvar_resultados(saida, anotado, eventos, totais, fotos):
    saida.mkdir(parents=True, exist_ok=True)
    anotado.replace(saida / "contagem_anotada.mp4")
    with (saida / "passagens.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=("periodo", "quadro", "tempo_s", "sentido", "tipo"))
        escritor.writeheader()
        escritor.writerows(eventos)
    with (saida / "contagem.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(("periodo", "veiculos"))
        escritor.writerows((("dia", totais["dia"]), ("noite", totais["noite"]),
                            ("total", sum(totais.values()))))
    for periodo, quadro in fotos.items():
        cv2.imwrite(str(saida / f"contagem_{periodo}.png"), quadro)
    print(f"Dia: {totais['dia']} | Noite: {totais['noite']} | Total: {sum(totais.values())}")
    print(f"Resultados: {saida}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sem-janela", action="store_true", help="Grava os resultados sem abrir o vídeo.")
    args = parser.parse_args()
    print("Analisando o vídeo...", flush=True)
    with TemporaryDirectory(dir=PASTA) as pasta:
        anotado = Path(pasta) / "contagem.mp4"
        eventos, totais, fotos = analisar(VIDEO, anotado)
        if not args.sem_janela:
            reproduzir(anotado)
        salvar_resultados(PASTA / "resultados", anotado, eventos, totais, fotos)


if __name__ == "__main__":
    main()
