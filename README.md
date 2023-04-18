*Bootcamp* de [Adalab](https://adalab.es/#), [Analistas de Datos](https://adalab.es/bootcamp-data/):

Desde el 18 de Enero hasta el 1 de Febrero 2023
[Giovanna Lozito](https://github.com/VannaLZ) y [Paula Fuente](https://github.com/paulafuenteg)

---

## Modulo 3: Machine Learning

---

### Índice

- [Regresion Lineal](#regresion-lineal)
    - [Regresion Lineal Estructura del repositorio](#regresion-lineal-estructura-del-repositorio)
    - [Regresion Lineal Librería](#regresion-lineal-librería)
- [Regresion Logistica](#regresion-logistica)
    - [Regresion Logistica Estructura del repositorio](#regresion-logistica-estructura-del-repositorio)
    - [Regresion Logistica Librería](#regresion-logistica-librería)

***
   
### **`Regresion Lineal`**

Empezamos con explorar el *Dataframe* que tenemos y decidir cual será nuestra variable respuesta: *Exposure*.   

Utilizamos el *DataFrame* [*Global Disaster Risk*](https://www.kaggle.com/datasets/tr1gg3rtrash/global-disaster-risk-index-time-series-dataset)

|Columna| Tipo de dato | Descripcion |
|-------|--------------|-------------|
|**Region**| String|	Nombre de la region.
|**WRI**	| Decimal |	*World Risk Score* (Puntuaciones de riesgo de las regiones)
|**Exposure**	| Decimal |	Riesgo/exposición a peligros naturales como terremotos, huracanes, inundaciones, sequías y aumento del nivel del mar.
|**Vulnerability**	| Decimal | Vulnerabilidad en función de la infraestructura, la nutrición, la situación de la vivienda y las condiciones del marco económico.
|**Susceptibility**	| Decimal |	Susceptibilidad según la infraestructura, la nutrición, la situación de la vivienda y las condiciones del marco económico.
|**Lack of Coping Capabilities**	| Decimal |	Preparación ante desastres, atención medica, seguridad social.
|**Lack of Adaptive Capacities**| Decimal |	Capacidades de adaptácion ante eventos naturales, cambio climático y otro desafíos.
|**Year**	| Decimal |	Años.
|**WRI Category**| String|	Categoria calculada en base al *WRI*.
|**Exposure Category**| String|	Categoria calculada en base al *Exposure*.
|**Vulnerability Categoy**| String|	Categoria calculada en base al *Vulnerability*.
|**Susceptibility Category**| String|	 Categoria calculada en base al *Susceptibility*.

---

### **Regresion Lineal Estructura del Repositorio**:
- **datos** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/datos)  
    Aquí encontramos todos los ficheros que hemos ido utilizando.  
    La serie de ficheros de  <span style="color:lightblue">Regresion Lineal</span> están nombrado *world_risk_index* y hay diferentes formados que hemos ido guardando a lo largo de nuestro *pair*.

- **deepl** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/deepl)  
Aquí encontramos los ficheros en lo que hemos realizado la traducción de la columna *region*.  
Enlace con toda la info sobre deepL [deepl-Python](https://github.com/DeepLcom/deepl-python).  
>:octocat::octocat: **ATENCÍON** :octocat::octocat:     
Se necesita el *token* para ejecutar el `jupiter` de *deepl*. 

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

---



### **Regresion Lineal Librería:**

```
#Traducción columna region
import deepL

# Tratamiento de datos
import numpy as np
import pandas as pd

# Test estadisticos
import researchpy as rp
from scipy import stats
from scipy.stats import kstest
from scipy.stats import levene
from scipy.stats import skew
from scipy.stats import kurtosistest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from sklearn.preprocessing import StandardScaler

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Transformación de los datos / modelado / evaluación / cross evaluacion
import math 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics

# Codificación de las variables numéricas
from sklearn.preprocessing import LabelEncoder # para realizar el Label Encoding 
from sklearn.preprocessing import OneHotEncoder  # para realizar el One-Hot Encoding

# Configuración warnings
import warnings
warnings.filterwarnings('once')
```
---

### **`Regresion Logistica`**


Empezamos con explorar el Dataframe que tenemos y decidir cual será nuestra variable respuesta: *fraud*:      
Utilizamos el *DataFrame* [Fraude de Tarjeta de Credito](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)

|Column| Type | Description |
|-------|--------------|-------------|
|distance_from_home| float64|	Distancia desde casa donde occurrió la transacción
|distance_from_last_transaction| float64|	Distancia desde donde occurrió la uñtima transacción  
|ratio_to_median_purchase_price| float64|	Ratio entre el precio de la transacción y el precio de la compra media
|repeat_retailer| float64|	¿La transacción se realizó desde el mismo vendidore/tienda? 
|used_chip| float64|	¿La transacción se realizó con el chip? 
|used_pin_number| float64|	¿La transacción se realizó utilizando el pin?  
|online_order | float64| ¿La transacción se realizó en internet? 
|fraud | float64| ¿La transacción es una fraude? 

---


### **Regresion Logistica Estructura del Repositorio**

- **datos** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/datos)    
    La serie de ficheros de Regresion Logistica están nombrados como resultado_fraude.

- **Regresion Logistica** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/Regresion%20Logistica)   
En los siguientes ficheros podemos encontar nuestro estudio sobre los datos, utilizando la metodologia EDA, averiguamos la distribución de los datos, los balanceamos, estandarizamos y utilizamos la matriz de correlación.  
Aplicamos ambos el *Decision Tree* y el *Random Forest*.

    - [Lecc01-EDA](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc01-EDA.ipynb)  
    - [Lecc02-Preparacion_Datos](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc02-Preparacion_Datos.ipynb)  
    - [Lecc03-Ajuste](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc03-Ajuste.ipynb)  
    - [Lecc04-Metricas](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc04-Metricas.ipynb)  
    - [Lecc05-Decision_Tree](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc05-Decision_Tree.ipynb)  
    - [Lecc06-Random_Forest](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc06-Random_Forest.ipynb)  

> :warning::warning: **ATENCÍON** :warning::warning:  
> Estos ultimos dos `jupiters` están ejecutado directamente en el `google colab`. 

---    

### **Regresion Logistica Librería:**

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

# Para separar los datos en train y test / matriz de confusión / Modelado 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , cohen_kappa_score, roc_curve,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#  Gestión de warnings
import warnings
warnings.filterwarnings("ignore")
``` 
---





