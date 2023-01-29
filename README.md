# DA-promoC-Mod3-sprint1-VannayPaula

*Bootcamp* de [Adalab](https://adalab.es/#), [Analistas de Datos](https://adalab.es/bootcamp-data/):

Desde el 18 de Enero hasta el 1 de Febrero 2023
[Giovanna Lozito](https://github.com/VannaLZ) y [Paula Fuente](https://github.com/paulafuenteg)

---

## Modulo 3: <span style="color:bluetblue"> Machine Learning</span>

### Índice

- [Regresion Lineal](#regresion-lineal)
    - [Regresion Lineal Estructura del repositorio](#rl-estructura-del-repositorio)
    - [Regresion Lineal Biblioteca](#rl-bibliotecas)
- [Regresion Logistica](#regresion-logistica)
    - Regresion Logistica Estructura del repositorio
    - Regresion Logistica Biblioteca

***
### `Regresion Lineal`

Empezamos con explorar el *Dataframe* que tenemos y decidir cual será nuestra variable respuesta.    

Utilizamos el DataFrame [*Global Disaster Risk*](https://www.kaggle.com/datasets/tr1gg3rtrash/global-disaster-risk-index-time-series-dataset)

|Columna| Tipo de dato | Descripcion |
|-------|--------------|-------------|
|**Region**| String|	Nombre de la region .
|**WRI**	| Decimal |	*World Risk Score* of the region.
|**Exposure**	| Decimal |	*Risk/exposure* to natural hazards such as earthquakes, hurricanes, floods, droughts, and sea ​​level rise.
|**Vulnerability**	| Decimal |	Vulnerability depending on infrastructure, nutrition, housing situation, and economic framework conditions.
|**Susceptibility**	| Decimal |	Susceptibility depending on infrastructure, nutrition, housing situation, and economic framework conditions.
|**Lack of Coping Capabilities**	| Decimal |	Coping capacities in dependence of governance, preparedness and early warning, medical care, and social and material security.
|**Lack of Adaptive Capacities**| Decimal |	Adaptive capacities related to coming natural events, climate change, and other challenges.
|**Year**	| Decimal |	Year data is being described.
|**WRI Category**| String|	WRI Category for the given WRI Score.
|**Exposure Category**| String|	Exposure Category for the given Exposure Score.
|**Vulnerability Categoy**| String|	Vulnerability Category for the given Vulnerability Score.
|**Susceptibility Category**| String|	Susceptibility Category for the given Susceptibility Score.


## **RL Estructura del repositorio**:
- **datos** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/datos)  
    Aquí encontramos todos los ficheros que hemos ido utilizando.  
    La serie de ficheros de  <span style="color:lightblue">Regresion Lineal</span> están nombrado *world_risk_index* y hay diferentes formados que hemos ido guardando a lo largo de nuestro *pair*.

- **deepl** - [Carpeta]()
Aquí encontramos los ficheros en lo que hemos realizado la traducción de la columna *region*.  
Enlace con toda la info sobre deepL [deepl-Python](https://github.com/DeepLcom/deepl-python).

- **Regresion Lineal** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/Regresion%20Lineal)  
En los siguientes ficheros podemos encontar nuestro estudio sobre los datos, utilizando la metodologia EDA, averiguamos si hay nulos, *outliers*, realizamos graficas.  
Averiguamos correlaciones, normalizamos, estandardizamos y aplicamos el *encoding* a los datos.  
Aplicamos la Regresion lineal, *Decision Tree* y *Random Tree*.  


    - [Lecc01-Intro_ML](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc01-Intro_ML.ipynb)
    - [Lecc02-Test_Estadisticos](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc02-Test_Estadisticos.ipynb)
    - [Lecc03-Correlación_Covarianza](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc03-Correlacion_Covarianza.ipynb)
    - [Lecc04-Asunciones](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc04-Asunciones.ipynb)
    - [Lecc05-Normalización](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc05-Normalizaci%C3%B3n.ipynb)
    - [Lecc06-Estandardizacion](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc06-Estandarizacion.ipynb)
    - [Lecc07-Anova](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc07-Anova.ipynb)
    - [Lecc08-Encoding](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc08-Encoding.ipynb)
    - [Lecc09-Regresion_lineal_Intro](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc09-Regresion_lineal_Intro.ipynb)
    - [Lecc10-Regresion_lineal_Metricas](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc10-Regresion-lineal_Metricas.ipynb)
    - [Lecc11-Decision_tree](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc11-Decision_Tree.ipynb)
    - [Lecc12-Forest_tree](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc12-Forest_Tree.ipynb)


### RL Bibliotecas:

```
#Traducción columna region
import deepL

# Tratamiento de datos
import numpy as np
import pandas as pd

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Modelado y evaluación
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

# Configuración warnings
import warnings
warnings.filterwarnings('once')
```


### <span style="color:blue">- Regresion Logistica</span>

---







