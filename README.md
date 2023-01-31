# DA-promoC-Mod3-sprint1-VannayPaula

*Bootcamp* de [Adalab](https://adalab.es/#), [Analistas de Datos](https://adalab.es/bootcamp-data/):

Desde el 18 de Enero hasta el 1 de Febrero 2023
[Giovanna Lozito](https://github.com/VannaLZ) y [Paula Fuente](https://github.com/paulafuenteg)

---

## Modulo 3: <span style="color:bluetblue"> Machine Learning</span>

### Índice

- [Regresion Lineal](#regresion-lineal)
    - [Regresion Lineal Estructura del repositorio](#regresion-lineal-estructura-del-repositorio)
    - [Regresion Lineal Biblioteca](#regresion-lineal-bibliotecas)
- [Regresion Logistica](#regresion-logistica)
    - [Regresion Logistica Estructura del repositorio]()
    - [Regresion Logistica Biblioteca](#regresion-logistica-biblioteca)

***
### **`Regresion Lineal`**

Empezamos con explorar el *Dataframe* que tenemos y decidir cual será nuestra variable respuesta.    

Utilizamos el *DataFrame* [*Global Disaster Risk*](https://www.kaggle.com/datasets/tr1gg3rtrash/global-disaster-risk-index-time-series-dataset)

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


### **Regresion Lineal Estructura del Repositorio**:
- **datos** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/datos)  
    Aquí encontramos todos los ficheros que hemos ido utilizando.  
    La serie de ficheros de  <span style="color:lightblue">Regresion Lineal</span> están nombrado *world_risk_index* y hay diferentes formados que hemos ido guardando a lo largo de nuestro *pair*.

- **deepl** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/deepl)
Aquí encontramos los ficheros en lo que hemos realizado la traducción de la columna *region*.  
Enlace con toda la info sobre deepL [deepl-Python](https://github.com/DeepLcom/deepl-python).

- **Regresion Lineal** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/Regresion%20Lineal)  
En los siguientes ficheros podemos encontar nuestro estudio sobre los datos, utilizando la metodologia EDA, averiguamos si hay nulos, *outliers*, realizamos graficas.  
Averiguamos correlaciones, normalizamos, estandardizamos y aplicamos el *encoding* a los datos.  
Aplicamos la Regresion lineal, *Decision Tree* y *Random Forest*.  


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
    - [Lecc12-Random_Forest](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc12-Random_Forest.ipynb)


### **Regresion Lineal Bibliotecas:**

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


### **`Regresion Logistica`**


Empezamos con explorar el Dataframe que tenemos y decidir cual será nuestra variable respuesta.  
Utilizamos el *DataFrame* [Fraude de Tarjeta de Credito](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)

|Column| Type | Description |
|-------|--------------|-------------|
|distance_from_home| float64|	The distance from home where the transaction happened
|distance_from_last_transaction| float64|	The distance from last transaction happened.
|ratio_to_median_purchase_price| float64|	Ratio of purchased price transaction to median purchase price.
|repeat_retailer| float64|	Is the transaction happened from same retailer. 
|used_chip| float64|	Is the transaction through chip (credit card)
|used_pin_number| float64|	Is the transaction happened by using PIN number. 
|online_order | float64| Is the transaction an online order.
|fraud | float64| Is the transaction fraudulent.   
---

### **Regresion Logistica Estructura del Repositorio**

En los siguientes ficheros podemos encontar nuestro estudio sobre los datos, utilizando la metodologia EDA, averiguamos si hay nulos, *outliers*, realizamos graficas.  
 
Aplicamos la Regresion logistica, *Decision Tree* y *Random Forest* y estos dos `jupiters` están ejecutado directamente en el `google colab`.

   - [Lecc01-EDA](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc01-EDA.ipynb)
    - [Lecc02-Preparacion_Datos](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc02-Preparacion_Datos.ipynb)
    - [Lecc03-Ajuste](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc03-Ajuste.ipynb)
    - [Lecc04-Metricas](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc04-Metricas.ipynb)
    - [Lecc05-Decision_Tree]()
    - [Lecc06-Random_Forest]()

### **Regresion Logistica Biblioteca:**

```# Tratamiento de datos
import numpy as np
import pandas as pd


# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns


# Estandarización variables numéricas y Codificación variables categóricas
from sklearn.preprocessing import StandardScaler

# Gestión datos desbalanceados
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

# Para separar los datos en train y test
from sklearn.model_selection import train_test_split

#  Gestión de warnings
import warnings
warnings.filterwarnings("ignore")
``` 





