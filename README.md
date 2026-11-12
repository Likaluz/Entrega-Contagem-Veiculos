# Contagem de veículos · dia e noite


**Integrantes e RMs**  
Diogo Witzel - RM552754

Lucas Domingues - RM553304

Victor Morelli - RM553338

O projeto conta veículos que cruzam as linhas marcadas em duas rodovias reais, em um único vídeo de 29 segundos. A contagem pode apoiar estudos de fluxo viário e mostrar o efeito da iluminação sobre uma câmera de monitoramento.

| Trecho | Duração | Veículos |
| --- | ---: | ---: |
| Dia | 15 s | 15 |
| Noite | 14 s | 6 |
| **Total** | **29 s** | **21** |

De dia, o detector registrou 12 carros, 2 caminhões e 1 ônibus. A revisão do quadro 77 mostra que o rótulo de ônibus corresponde a um caminhão. A passagem foi contada. À noite, o sistema registra o tipo genérico `veiculo` porque conta pelas luzes.

O [relatório do projeto](relatorio.pdf) reúne o contexto, o método, os resultados e as limitações. O [notebook de análise](analise.ipynb) lê os registros e reproduz a comparação com a execução anterior.

## Executar

Na pasta do projeto, crie um ambiente com Python 3.12 e instale as dependências pelo PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Execute o script:

```powershell
.\.venv\Scripts\python.exe contar_veiculos.py
```

Mantenha o vídeo `trafego_dia_noite.mp4` e o modelo `yolov5n.onnx` ao lado do script. O programa analisa o vídeo inteiro e abre a reprodução com o contador. Pressione **Esc** ou feche a janela para mostrar o total no console e gravar os arquivos em `resultados/`. Use `--sem-janela` para gerar os mesmos arquivos direto pelo terminal.

Para reexecutar o notebook, instale o JupyterLab no mesmo ambiente, inicie-o na raiz do projeto, abra `analise.ipynb` e execute as células em ordem:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyterlab
```

## Métricas e evidências

| Trecho | Referência YOLOv5m | Execução YOLOv5n | Diferença absoluta | Pares em até 3 quadros |
| --- | ---: | ---: | ---: | ---: |
| Dia | 15 | 15 | 0 | 15 |
| Noite | 6 | 6 | 0 | 6 |
| Total | 21 | 21 | 0 | 21 |

A referência é uma execução anterior no mesmo vídeo. O notebook pareia as passagens por ordem dentro de cada período e sentido. A maior diferença foi de três quadros, ou 0,25 s a 12 quadros/s. A tabela mede concordância entre as duas execuções. O quadro 77 documenta um erro de classificação, e o estudo não estimou acurácia com uma base anotada independente.

- [Gráfico de contagens e linha do tempo](resultados/grafico_contagens.png).
- [Indicadores por período](resultados/metricas.csv) e [comparação passagem a passagem](resultados/comparacao_passagens.csv).
- [Totais atuais](resultados/contagem.csv), [passagens atuais](resultados/passagens.csv) e [passagens da referência](resultados/referencia_yolov5m.csv).
- [Vídeo anotado](resultados/contagem_anotada.mp4), [última passagem diurna](resultados/contagem_dia.png), [última passagem noturna](resultados/contagem_noite.png) e [erro de classificação no quadro 77](resultados/evidencia_classificacao.png).

