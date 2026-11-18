# Proyecto de Ciencia de Datos — Respuesta sanitaria de Bolivia frente al COVID-19

Autor: Henry Pilco Cussy · Universidad Privada Franz Tamayo · Ingeniería de Sistemas

**Resultados en línea:** https://pxndxher.github.io/covid-bolivia/

## Estructura
- `datos/` — archivos de entrada (sin modificar)
  - `mun_covid_se30.csv`: casos acumulados por municipio de Bolivia, semanas epidemiológicas 20 a 30 de 2020.
  - `owid-covid-data.csv`: base de COVID-19 de Our World in Data, con datos hasta el 19/09/2020.
- `codigo/`
  - `eda_analisis.py`: análisis exploratorio (sección 9 del informe).
  - `analisis_modelos.py`: modelado (secciones 10 a 13): k-means, series de tiempo, índice de presión sanitaria y clasificación de riesgo municipal.
- `figuras/` — figuras generadas por los scripts.
- `resultados/` — métricas y estadísticas en formato JSON.
- `index.html` — página web con los resultados de los modelos (publicada con GitHub Pages).

## Cómo ejecutar
Requisitos: Python 3.10+, pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
cd codigo
DATA=../datos FIG=../figuras python eda_analisis.py
DATA=../datos FIG=../figuras python analisis_modelos.py
```
Los procedimientos aleatorios usan la semilla 42, por lo que los resultados son reproducibles.
