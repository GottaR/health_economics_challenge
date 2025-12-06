# ============================================================================
# FEATURE ENGINEERING PARA HEALTH ECONOMICS
# Universidad del Oeste (UNO) - Aplicaciones en Ciencia de Datos
# ============================================================================
# Este script realiza la ingeniería de características (Feature Engineering)
# sobre el dataset de health economics para predecir el gasto de bolsillo
# en salud (hf3_ppp_pc) para el año 2022.
#
# El Feature Engineering es el proceso de crear nuevas variables predictoras
# a partir de las variables originales del dataset, para mejorar el desempeño
# del modelo de Machine Learning.
# ============================================================================

# ----------------------------------------------------------------------------
# LIBRERÍAS NECESARIAS
# ----------------------------------------------------------------------------
require("data.table")  # Manejo eficiente de datos
require("Rcpp")        # Para funciones en C++ (mayor velocidad)
require("rlist")       # Manejo de listas
require("yaml")        # Leer archivos de configuración YAML
library(dplyr)         # Manipulación de datos
library(stringr)       # Manejo de strings
library(lubridate)     # Manejo de fechas

require("lightgbm")    # Algoritmo de Machine Learning
require("randomForest") # Para imputar valores faltantes

# ----------------------------------------------------------------------------
# FUNCIONES DE UTILIDAD
# ----------------------------------------------------------------------------

# Reporta la cantidad de campos (columnas) del dataset
# Útil para trackear cómo va creciendo el dataset con nuevas variables
ReportarCampos <- function(dataset) {
  cat("La cantidad de campos es", ncol(dataset), "\n")
}

# ----------------------------------------------------------------------------
# AGREGAR VARIABLE CÍCLICA DE AÑO
# ----------------------------------------------------------------------------
# Agrega una variable que representa un ciclo temporal basado en el año.
# Esto ayuda al modelo a capturar patrones cíclicos en el tiempo.
#
# Por ejemplo, si tenemos datos desde 2000 hasta 2020, esta función
# crea una variable que va de 1 a 10 en un ciclo repetitivo.
# Esto es similar a cómo el "mes" captura estacionalidad mensual,
# pero aplicado a años con un ciclo de 10 años.

AgregarMes <- function(dataset) {
  gc()  # Garbage collection para liberar memoria

  # Crear variable cíclica global por año
  # ((year - min(year)) %% 10) + 1 genera un ciclo de 1 a 10
  # Todos los países en el mismo año tendrán el mismo valor de ciclo
  dataset[, year_cycle := ((year - min(year, na.rm = TRUE)) %% 10) + 1]

  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# ELIMINAR VARIABLES CON DATA DRIFTING
# ----------------------------------------------------------------------------
# Elimina variables que se sospecha causan "data drifting".
# Data drifting ocurre cuando la distribución de una variable cambia
# significativamente a lo largo del tiempo, haciendo que el modelo
# aprenda patrones que no se mantendrán en el futuro.
#
# Parámetros:
#   - dataset: Dataset a modificar
#   - variables: Vector con nombres de variables a eliminar

DriftEliminar <- function(dataset, variables) {
  gc()
  dataset[, c(variables) := NULL]
  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# CREAR DUMMIES PARA VALORES FALTANTES
# ----------------------------------------------------------------------------
# Para cada variable que tiene valores faltantes (NA), crea una nueva
# variable dummy (0/1) indicando si el valor era NA o no.
#
# ¿Por qué es útil? Porque "falta de dato" puede ser información valiosa:
# - Si un país no reporta cierta métrica, puede indicar algo sobre ese país
# - El modelo puede aprender patrones asociados a la ausencia de datos
#
# Ejemplo: Si la variable "gasto_salud" tiene NAs, se crea "gasto_salud_isNA"
#          donde 1 = el valor estaba faltante, 0 = el valor estaba presente

DummiesNA <- function(dataset) {
  gc()

  # Contar cantidad de nulos por columna en el año presente
  nulos <- colSums(is.na(dataset[year %in% PARAMS$feature_engineering$const$presente]))

  # Identificar columnas que tienen al menos un nulo
  colsconNA <- names(which(nulos > 0))

  # Crear variables dummy para cada columna con NAs
  # .SD significa "Subset of Data" - se aplica la función a cada columna seleccionada
  # .SDcols especifica sobre qué columnas aplicar la función
  dataset[, paste0(colsconNA, "_isNA") := lapply(.SD, is.na),
          .SDcols = colsconNA]

  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# AGREGAR VARIABLES MANUALES
# ----------------------------------------------------------------------------
# Esta es la sección donde TÚ como alumno debes desplegar tu creatividad
# e ingenio para crear nuevas variables que ayuden a predecir el gasto
# de bolsillo en salud (hf3_ppp_pc).
#
# Las variables creadas aquí son controladas por parámetros del archivo YAML,
# lo que permite activarlas o desactivarlas fácilmente.
#
# EJEMPLO DE VARIABLE INCLUIDA:
# - YearsSinceFirst: Años transcurridos desde el primer registro válido
#                    de hf3_ppp_pc para cada país y región.
#
# IDEAS PARA NUEVAS VARIABLES (tú puedes implementarlas):
# - Ratios entre variables (ej: gasto_salud / PIB)
# - Interacciones entre variables (ej: expectativa_vida * gasto_per_capita)
# - Transformaciones logarítmicas o potencias
# - Variables dummy para eventos específicos (crisis económicas, pandemias)
# - Agregaciones por región o nivel de ingreso

AgregarVariables <- function(dataset) {
  gc()

  # INICIO de la sección donde debes crear tus propias variables

  # EJEMPLO: Calcular años desde el primer registro válido
  # Esta variable captura cuánto tiempo lleva cada país reportando datos
  # de gasto de bolsillo en salud (hf3_ppp_pc).

  # Paso 1: Para cada combinación de región y país, encontrar el primer año
  #         donde hf3_ppp_pc es mayor a 0 (primer dato válido)
  dataset[hf3_ppp_pc > 0, FirstYear := min(year, na.rm = TRUE),
          by = .(region, `Country Code`)]

  # Paso 2: Propagar el valor de FirstYear a todas las filas del mismo país/región
  #         nafill rellena los NAs: "locf" = last observation carried forward
  #                                "nocb" = next observation carried backward
  dataset[, FirstYear := nafill(FirstYear, type = "locf"),
          by = .(region, `Country Code`)]
  dataset[, FirstYear := nafill(FirstYear, type = "nocb"),
          by = .(region, `Country Code`)]

  # Paso 3: Calcular cuántos años han pasado desde el primer año con datos
  dataset[, YearsSinceFirst := year - FirstYear]

  # --- AGREGA TUS PROPIAS VARIABLES AQUÍ ---
  # 2.1 Índice de Salud Materno-Infantil
  if(all(c("SH.DTH.IMRT", "SH.STA.MMRT", "SH.DTH.MORT") %in% names(dataset))) {
    dataset[, maternal_child_health_index := 
              (SH.DTH.IMRT + SH.STA.MMRT/10 + SH.DTH.MORT) / 3]
    cat("✓ maternal_child_health_index\n")
  }
  
  # 2.2 Índice de Mortalidad Adulta (promedio hombres y mujeres)
  if(all(c("SH.DYN.MORT.FE", "SH.DYN.MORT.MA") %in% names(dataset))) {
    dataset[, adult_mortality_index := (SH.DYN.MORT.FE + SH.DYN.MORT.MA) / 2]
    cat("✓ adult_mortality_index\n")
  }
  
  # 2.3 Brecha de Género en Mortalidad Adulta
  if(all(c("SH.DYN.MORT.MA", "SH.DYN.MORT.FE") %in% names(dataset))) {
    dataset[, mortality_gender_gap := SH.DYN.MORT.MA - SH.DYN.MORT.FE]
    cat("✓ mortality_gender_gap\n")
  }
  
  # 2.4 Cobertura de Inmunización Promedio
  if(all(c("SH.IMM.IDPT", "SH.IMM.MEAS") %in% names(dataset))) {
    dataset[, immunization_coverage := (SH.IMM.IDPT + SH.IMM.MEAS) / 2]
    cat("✓ immunization_coverage\n")
  }
  
  # 2.5 Índice de Sistema de Salud (inverso de mortalidad + expectativa vida)
  if(all(c("SP.DYN.LE00.IN", "SH.DTH.IMRT") %in% names(dataset))) {
    dataset[, health_system_index := SP.DYN.LE00.IN / (SH.DTH.IMRT + 1)]
    cat("✓ health_system_index\n")
  }
  # Ejemplo comentado (no se ejecuta):
  # if(PARAMS$feature_engineering$param$health_ratios) {
  #   dataset[, mi_ratio := variable1 / (variable2 + 1)]  # +1 para evitar división por 0
  # }

  # --- VÁLVULAS DE SEGURIDAD ---
  # Estas secciones protegen al dataset de valores problemáticos

  # VÁLVULA 1: Detectar y corregir valores infinitos (Inf, -Inf)
  # Los infinitos aparecen típicamente por divisiones por cero (ej: x/0 = Inf)
  infinitos <- lapply(names(dataset), function(.name) dataset[, sum(is.infinite(get(.name)))])
  infinitos_qty <- sum(unlist(infinitos))

  if (infinitos_qty > 0) {
    cat("ATENCIÓN: hay", infinitos_qty, "valores infinitos en tu dataset. Serán pasados a NA\n")
    dataset[mapply(is.infinite, dataset)] <<- NA
  }

  # VÁLVULA 2: Detectar y corregir valores NaN (Not a Number)
  # Los NaN aparecen por operaciones indeterminadas como 0/0
  # Decisión polémica: los pasamos a 0 (puedes modificar según tu criterio)
  nans <- lapply(names(dataset), function(.name) dataset[, sum(is.nan(get(.name)))])
  nans_qty <- sum(unlist(nans))

  if (nans_qty > 0) {
    cat("ATENCIÓN: hay", nans_qty, "valores NaN (0/0) en tu dataset. Serán pasados arbitrariamente a 0\n")
    cat("Si no te gusta la decisión, modifica a gusto el programa!\n\n")
    dataset[mapply(is.nan, dataset)] <<- 0
  }

  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# CALCULAR LAGS (VALORES RETRASADOS)
# ----------------------------------------------------------------------------
# Esta función crea variables "lag" (valores de años anteriores) para cada
# variable especificada, y opcionalmente calcula los "deltas" (diferencias).
#
# ¿Por qué son útiles los lags?
# En series de tiempo, el pasado ayuda a predecir el futuro.
# Por ejemplo: El gasto en salud del año pasado (lag_1) es probablemente
# un buen predictor del gasto de este año.
#
# Parámetros:
#   - cols: Vector de nombres de columnas para las cuales crear lags
#   - nlag: Número de períodos hacia atrás (ej: nlag=1 es el año anterior)
#   - deltas: Si TRUE, también calcula la diferencia entre valor actual y lag
#
# Ejemplo con nlag=1:
#   Si tenemos: 2019: GDP=1000, 2020: GDP=1100
#   Se crea:    2020: GDP_lag1=1000, GDP_delta1=100 (1100-1000)
#
# IMPORTANTE: Esta función asume que el dataset está ordenado por
# Country Code y year (ascendente).

Lags <- function(cols, nlag, deltas) {
  gc()
  sufijo <- paste0("_lag", nlag)

  # Crear variables lag usando la función shift de data.table
  # shift(x, n, NA, "lag") toma el valor de n posiciones atrás
  # by = `Country Code` asegura que no se mezclen datos de diferentes países
  dataset[, paste0(cols, sufijo) := shift(.SD, nlag, NA, "lag"),
          by = `Country Code`,
          .SDcols = cols]

  # Si deltas=TRUE, calcular la diferencia entre valor actual y lag
  if (deltas) {
    sufijodelta <- paste0("_delta", nlag)

    for (vcol in cols) {
      # Delta = Valor actual - Valor lag
      # Ejemplo: GDP_delta1 = GDP - GDP_lag1 (crecimiento del PIB)
      dataset[, paste0(vcol, sufijodelta) := get(vcol) - get(paste0(vcol, sufijo))]
    }
  }

  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# FUNCIÓN C++ PARA CALCULAR ESTADÍSTICAS DE VENTANA
# ----------------------------------------------------------------------------
# Esta función en C++ calcula estadísticas sobre ventanas móviles de tiempo.
# Está escrita en C++ para mayor velocidad de procesamiento.
#
# Para cada observación, calcula sobre una "ventana" de años anteriores:
# - Tendencia (pendiente de regresión lineal)
# - Mínimo
# - Máximo
# - Promedio
# - Lag del período anterior
#
# ¿Por qué en C++? Porque R es lento para bucles. Esta función procesa
# miles de observaciones mucho más rápido que código equivalente en R.

cppFunction('NumericVector fhistC(NumericVector pcolumna, IntegerVector pdesde)
{
  /* Vectores para almacenar datos de la regresión */
  double  x[100];  // Posiciones temporales
  double  y[100];  // Valores de la variable

  int n = pcolumna.size();
  NumericVector out(5*n);  // Vector de salida: 5 estadísticas por cada observación

  for(int i = 0; i < n; i++)
  {
    // Calcular lag (valor del período anterior)
    if(pdesde[i]-1 < i)  out[i + 4*n] = pcolumna[i-1];
    else                 out[i + 4*n] = NA_REAL;

    int libre = 0;      // Contador de valores válidos
    int xvalor = 1;     // Posición temporal

    // Recopilar valores de la ventana (desde pdesde[i] hasta i)
    for(int j = pdesde[i]-1; j <= i; j++)
    {
      double a = pcolumna[j];

      // Solo incluir valores no-NA
      if(!R_IsNA(a))
      {
        y[libre] = a;
        x[libre] = xvalor;
        libre++;
      }

      xvalor++;
    }

    /* Si hay al menos dos valores, calcular estadísticas */
    if(libre > 1)
    {
      double xsum  = x[0];
      double ysum  = y[0];
      double xysum = xsum * ysum;
      double xxsum = xsum * xsum;
      double vmin  = y[0];
      double vmax  = y[0];

      for(int h = 1; h < libre; h++)
      {
        xsum  += x[h];
        ysum  += y[h];
        xysum += x[h] * y[h];
        xxsum += x[h] * x[h];

        if(y[h] < vmin) vmin = y[h];
        if(y[h] > vmax) vmax = y[h];
      }

      // Calcular pendiente de regresión lineal (tendencia)
      out[i]        = (libre*xysum - xsum*ysum) / (libre*xxsum - xsum*xsum);
      out[i + n]    = vmin;      // Mínimo
      out[i + 2*n]  = vmax;      // Máximo
      out[i + 3*n]  = ysum / libre;  // Promedio
    }
    else
    {
      // Si no hay suficientes valores, retornar NA
      out[i]       = NA_REAL;
      out[i + n]   = NA_REAL;
      out[i + 2*n] = NA_REAL;
      out[i + 3*n] = NA_REAL;
    }
  }

  return out;
}')

# ----------------------------------------------------------------------------
# CALCULAR TENDENCIAS Y ESTADÍSTICAS MÓVILES
# ----------------------------------------------------------------------------
# Calcula estadísticas sobre ventanas móviles de N años hacia atrás.
# Esto captura tendencias y patrones temporales en los datos.
#
# Parámetros:
#   - dataset: Dataset a modificar
#   - cols: Columnas sobre las cuales calcular estadísticas
#   - ventana: Tamaño de la ventana en años (ej: 6 = últimos 6 años)
#   - tendencia: Si TRUE, calcula la pendiente (¿está creciendo o cayendo?)
#   - minimo: Si TRUE, calcula el valor mínimo en la ventana
#   - maximo: Si TRUE, calcula el valor máximo en la ventana
#   - promedio: Si TRUE, calcula el promedio en la ventana
#   - ratioavg: Si TRUE, calcula valor_actual / promedio_ventana
#   - ratiomax: Si TRUE, calcula valor_actual / máximo_ventana
#
# Ejemplo con ventana=3 y promedio=TRUE:
#   Para el año 2020, calcula el promedio de 2018, 2019 y 2020

TendenciaYmuchomas <- function(dataset, cols, ventana = 6, tendencia = TRUE,
                                minimo = TRUE, maximo = TRUE, promedio = TRUE,
                                ratioavg = FALSE, ratiomax = FALSE) {
  gc()

  # Cantidad de años hacia atrás que se usan para la historia
  ventana_regresion <- ventana
  last <- nrow(dataset)

  # Crear vector que indica el inicio de cada ventana
  # Esto acelera el procesamiento al calcularlo una sola vez
  vector_ids <- dataset$`Country Code`

  # Vector que indica desde qué fila comenzar la ventana para cada observación
  vector_desde <- seq(-ventana_regresion + 2, nrow(dataset) - ventana_regresion + 1)
  vector_desde[1:ventana_regresion] <- 1

  # Ajustar ventanas al cambiar de país (no mezclar datos de diferentes países)
  for (i in 2:last) {
    if (vector_ids[i - 1] != vector_ids[i]) {
      vector_desde[i] <- i  # Nueva ventana al cambiar de país
    }
  }

  # Propagar el inicio de ventana hacia adelante dentro de cada país
  for (i in 2:last) {
    if (vector_desde[i] < vector_desde[i - 1]) {
      vector_desde[i] <- vector_desde[i - 1]
    }
  }

  # Calcular estadísticas para cada campo especificado
  for (campo in cols) {
    # Llamar a la función C++ que calcula todas las estadísticas
    nueva_col <- fhistC(dataset[, get(campo)], vector_desde)

    # Crear nuevas columnas según los parámetros activados
    if (tendencia) {
      dataset[, paste0(campo, "_tend", ventana) := nueva_col[(0*last + 1):(1*last)]]
    }
    if (minimo) {
      dataset[, paste0(campo, "_min", ventana) := nueva_col[(1*last + 1):(2*last)]]
    }
    if (maximo) {
      dataset[, paste0(campo, "_max", ventana) := nueva_col[(2*last + 1):(3*last)]]
    }
    if (promedio) {
      dataset[, paste0(campo, "_avg", ventana) := nueva_col[(3*last + 1):(4*last)]]
    }
    if (ratioavg) {
      dataset[, paste0(campo, "_ratioavg", ventana) :=
                get(campo) / nueva_col[(3*last + 1):(4*last)]]
    }
    if (ratiomax) {
      dataset[, paste0(campo, "_ratiomax", ventana) :=
                get(campo) / nueva_col[(2*last + 1):(3*last)]]
    }
  }

  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# SELECCIÓN DE VARIABLES POR IMPORTANCIA (CANARITOS)
# ----------------------------------------------------------------------------
# Esta función elimina variables poco importantes usando el "método del canarito".
#
# ¿Qué es el método del canarito?
# Se agregan variables aleatorias (canaritos) al dataset. Luego se entrena
# un modelo y se mide la importancia de cada variable. Las variables REALES
# que tienen menor importancia que los canaritos (variables aleatorias) son
# eliminadas, ya que no aportan información útil.
#
# Es como los canarios en las minas de carbón: si el canarito (variable aleatoria)
# tiene más importancia que una variable real, esa variable real es "tóxica"
# y debe ser eliminada.
#
# Parámetros:
#   - canaritos_ratio: Ratio de variables canarito a agregar
#                      0.2 = agregar 20% de canaritos respecto al total de variables

GVEZ <- 1  # Contador global para trackear ejecuciones

CanaritosImportancia <- function(canaritos_ratio = 0.2) {
  gc()
  ReportarCampos(dataset)

  # Configuración de períodos para train y validation
  canaritos_year_end <- PARAMS$feature_engineering$const$canaritos_year_end

  # Agregar variables canarito (aleatorias uniformes entre 0 y 1)
  for (i in 1:(ncol(dataset) * canaritos_ratio)) {
    dataset[, paste0("canarito", i) := runif(nrow(dataset))]
  }

  # Identificar campos útiles (excluir identificadores y clase)
  campos_buenos <- setdiff(colnames(dataset),
                           c("Country Code", "year", PARAMS$feature_engineering$const$clase))

  # Crear conjunto de entrenamiento: 10% de datos entre años específicos
  azar <- runif(nrow(dataset))
  dataset[, entrenamiento :=
            year >= PARAMS$feature_engineering$const$canaritos_year_start &
            year < canaritos_year_end &
            azar < 0.10]

  # Preparar dataset para LightGBM (train)
  dtrain <- lgb.Dataset(
    data = data.matrix(dataset[entrenamiento == TRUE, campos_buenos, with = FALSE]),
    label = dataset[entrenamiento == TRUE, get(PARAMS$feature_engineering$const$clase)],
    free_raw_data = FALSE
  )

  # Preparar dataset de validación
  canaritos_year_valid <- PARAMS$feature_engineering$const$canaritos_year_valid

  dvalid <- lgb.Dataset(
    data = data.matrix(dataset[year == canaritos_year_valid, campos_buenos, with = FALSE]),
    label = dataset[year == canaritos_year_valid, get(PARAMS$feature_engineering$const$clase)],
    free_raw_data = FALSE
  )

  # Configurar parámetros de LightGBM
  param <- list(
    objective = "regression",
    metric = "rmse",
    first_metric_only = TRUE,
    boost_from_average = TRUE,
    feature_pre_filter = FALSE,
    verbosity = -100,
    seed = 999983,
    max_depth = -1,
    min_gain_to_split = 0.0,
    lambda_l1 = 0.0,
    lambda_l2 = 0.0,
    max_bin = 1023,
    num_iterations = 500,
    force_row_wise = TRUE,
    learning_rate = 0.065,
    feature_fraction = 1.0,
    min_data_in_leaf = 260,
    num_leaves = 60,
    early_stopping_rounds = 50
  )

  # Entrenar modelo para medir importancia de variables
  modelo <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalid),
    param = param,
    verbose = -100
  )

  # Obtener tabla de importancia de variables
  tb_importancia <- lgb.importance(model = modelo)
  tb_importancia[, pos := .I]  # Agregar posición (ranking)

  GVEZ <<- GVEZ + 1

  # Calcular umbral: mediana de posición de canaritos + 2 desviaciones estándar
  # Variables por debajo de este umbral se consideran más importantes que los canaritos
  umbral <- tb_importancia[Feature %like% "canarito", median(pos) + 2*sd(pos)]

  # Seleccionar variables útiles (más importantes que canaritos)
  col_utiles <- tb_importancia[pos < umbral & !(Feature %like% "canarito"), Feature]

  # Asegurar que campos esenciales siempre se mantengan
  col_utiles <- unique(c(col_utiles,
                        c("Country Code", "year",
                          PARAMS$feature_engineering$const$clase, "year_cycle")))

  # Identificar y eliminar columnas inútiles
  col_inutiles <- setdiff(colnames(dataset), col_utiles)
  dataset[, (col_inutiles) := NULL]

  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# CREAR RANKINGS RELATIVOS
# ----------------------------------------------------------------------------
# Para cada variable especificada, crea una nueva variable "_rank" que
# indica el ranking relativo de ese país en ese año.
#
# ¿Por qué es útil?
# Los rankings son robustos a outliers y capturan la posición relativa
# del país respecto a otros países en el mismo año.
#
# Ejemplo:
#   Si en 2020 un país tiene el 3er mayor PIB de 100 países,
#   PIB_rank = 3/100 = 0.03
#
# Parámetros:
#   - cols: Vector de nombres de columnas para las cuales crear rankings

Rankeador <- function(cols) {
  gc()
  sufijo <- "_rank"

  for (vcol in cols) {
    # frank() calcula el ranking, ties.method="random" resuelve empates al azar
    # Dividir por .N (cantidad total) normaliza entre 0 y 1
    # by = year asegura que el ranking se calcula dentro de cada año
    dataset[, paste0(vcol, sufijo) := frank(get(vcol), ties.method = "random") / .N,
            by = year]
  }

  ReportarCampos(dataset)
}

# ============================================================================
# PROGRAMA PRINCIPAL - PIPELINE DE FEATURE ENGINEERING
# ============================================================================
# A partir de aquí comienza la ejecución del script.
# El pipeline sigue estos pasos principales:
#
# 1. Cargar el dataset original
# 2. Ordenar por Country Code y year
# 3. Crear la variable objetivo (clase) con lead
# 4. Agregar variables de ciclo temporal
# 5. Crear dummies para valores faltantes
# 6. Agregar variables manuales creadas por el alumno
# 7. Calcular lags (valores retrasados)
# 8. Calcular tendencias y estadísticas móviles
# 9. Crear rankings relativos
# 10. Aplicar filtros de importancia (canaritos)
# 11. Limpiar valores problemáticos (NaN, infinitos)
# 12. Guardar dataset procesado
# ============================================================================

# Cargar el dataset original
setwd(PARAMS$environment$base_dir)
setwd(PARAMS$environment$data_dir)
nom_arch <- PARAMS$feature_engineering$files$input$dentrada
dataset <- fread(nom_arch)

# Ordenar el dataset por Country Code y year
# Esto es ESENCIAL para el correcto funcionamiento de lags y ventanas móviles
setorderv(dataset, PARAMS$feature_engineering$const$campos_sort)

# ----------------------------------------------------------------------------
# CREAR VARIABLE OBJETIVO (CLASE)
# ----------------------------------------------------------------------------
# La variable objetivo es lo que queremos predecir.
# En este caso: hf3_ppp_pc del año siguiente (lead = 1 año hacia adelante)
#
# Ejemplo:
#   Para el año 2020, la clase será el valor de hf3_ppp_pc de 2021
#   Para el año 2021, la clase será el valor de hf3_ppp_pc de 2022 (que NO conocemos)

# Paso 1: Copiar la variable original a una nueva columna llamada "clase"
dataset[, PARAMS$feature_engineering$const$clase :=
          get(PARAMS$feature_engineering$const$origen_clase),
        by = c("region", "Country Code")]

# Paso 2: Aplicar lead (desplazar hacia atrás) según orden_lead
# shift(..., type="lead") toma el valor de N posiciones hacia adelante
# Si orden_lead=1, toma el valor del año siguiente
dataset[, PARAMS$feature_engineering$const$clase :=
          shift(get(PARAMS$feature_engineering$const$clase),
                n = PARAMS$feature_engineering$const$orden_lead,
                type = "lead"),
        by = c("region", "Country Code")]

# ----------------------------------------------------------------------------
# APLICAR TRANSFORMACIONES SEGÚN CONFIGURACIÓN YAML
# ----------------------------------------------------------------------------

# Agregar ciclo de año
AgregarMes(dataset)

# Eliminar variables con drift (si están especificadas en el YAML)
if (length(PARAMS$feature_engineering$param$variablesdrift) > 0) {
  DriftEliminar(dataset, PARAMS$feature_engineering$param$variablesdrift)
}

# Crear dummies para valores faltantes
# IMPORTANTE: Esta línea debe ir ANTES de cualquier corrección de NAs
if (PARAMS$feature_engineering$param$dummiesNA) {
  DummiesNA(dataset)
}

# Agregar variables manuales creadas por el alumno
if (PARAMS$feature_engineering$param$variablesmanuales) {
  AgregarVariables(dataset)
}

# Eliminar la variable objetivo original (ya tenemos "clase")
dataset[, PARAMS$feature_engineering$const$origen_clase := NULL]

# ----------------------------------------------------------------------------
# CALCULAR LAGS Y TENDENCIAS
# ----------------------------------------------------------------------------
# Identificar columnas "lagueables" (todas excepto campos fijos como IDs)
cols_lagueables <- copy(setdiff(colnames(dataset),
                                PARAMS$feature_engineering$const$campos_fijos))

# Aplicar TendenciaYmuchomas según configuración YAML
# Se pueden configurar múltiples ventanas (ej: 2 años, 3 años, 5 años, etc.)
for (i in 1:length(PARAMS$feature_engineering$param$tendenciaYmuchomas$correr)) {
  if (PARAMS$feature_engineering$param$tendenciaYmuchomas$correr[i]) {

    # Si acumulavars=TRUE, incluir también variables recién creadas
    if (PARAMS$feature_engineering$param$acumulavars) {
      cols_lagueables <- setdiff(colnames(dataset),
                                 PARAMS$feature_engineering$const$campos_fijos)
    }

    cols_lagueables <- intersect(colnames(dataset), cols_lagueables)

    # Calcular tendencias y estadísticas móviles
    TendenciaYmuchomas(
      dataset,
      cols = cols_lagueables,
      ventana = PARAMS$feature_engineering$param$tendenciaYmuchomas$ventana[i],
      tendencia = PARAMS$feature_engineering$param$tendenciaYmuchomas$tendencia[i],
      minimo = PARAMS$feature_engineering$param$tendenciaYmuchomas$minimo[i],
      maximo = PARAMS$feature_engineering$param$tendenciaYmuchomas$maximo[i],
      promedio = PARAMS$feature_engineering$param$tendenciaYmuchomas$promedio[i],
      ratioavg = PARAMS$feature_engineering$param$tendenciaYmuchomas$ratioavg[i],
      ratiomax = PARAMS$feature_engineering$param$tendenciaYmuchomas$ratiomax[i]
    )

    # Aplicar filtro de canaritos si está configurado
    if (PARAMS$feature_engineering$param$tendenciaYmuchomas$canaritos[i] > 0) {
      CanaritosImportancia(
        canaritos_ratio = unlist(PARAMS$feature_engineering$param$tendenciaYmuchomas$canaritos[i])
      )
    }
  }
}

# Aplicar Lags según configuración YAML
for (i in 1:length(PARAMS$feature_engineering$param$lags$correr)) {
  if (PARAMS$feature_engineering$param$lags$correr[i]) {

    # Si acumulavars=TRUE, incluir variables recién creadas
    if (PARAMS$feature_engineering$param$acumulavars) {
      cols_lagueables <- setdiff(colnames(dataset),
                                 PARAMS$feature_engineering$const$campos_fijos)
    }

    cols_lagueables <- intersect(colnames(dataset), cols_lagueables)

    # Calcular lags
    Lags(
      cols_lagueables,
      PARAMS$feature_engineering$param$lags$lag[i],
      PARAMS$feature_engineering$param$lags$delta[i]
    )

    # Aplicar filtro de canaritos si está configurado
    if (PARAMS$feature_engineering$param$lags$canaritos[i] > 0) {
      CanaritosImportancia(
        canaritos_ratio = unlist(PARAMS$feature_engineering$param$lags$canaritos[i])
      )
    }
  }
}

# Actualizar cols_lagueables si acumulavars está activo
if (PARAMS$feature_engineering$param$acumulavars) {
  cols_lagueables <- setdiff(colnames(dataset),
                             PARAMS$feature_engineering$const$campos_fijos)
}

# ----------------------------------------------------------------------------
# CREAR RANKINGS
# ----------------------------------------------------------------------------
# Si rankeador está activo en el YAML, crear rankings para todas las variables
if (PARAMS$feature_engineering$param$rankeador) {

  if (PARAMS$feature_engineering$param$acumulavars) {
    cols_lagueables <- setdiff(colnames(dataset),
                               PARAMS$feature_engineering$const$campos_fijos)
  }

  cols_lagueables <- intersect(colnames(dataset), cols_lagueables)

  # Reordenar por year y Country Code para calcular rankings por año
  setorderv(dataset, PARAMS$feature_engineering$const$campos_rsort)
  Rankeador(cols_lagueables)

  # Volver al orden original (Country Code, year)
  setorderv(dataset, PARAMS$feature_engineering$const$campos_sort)
}

# Aplicar filtro final de canaritos si está configurado
if (PARAMS$feature_engineering$param$canaritos_final > 0) {
  CanaritosImportancia(canaritos_ratio = PARAMS$feature_engineering$param$canaritos_final)
}

# ----------------------------------------------------------------------------
# LIMPIEZA FINAL
# ----------------------------------------------------------------------------

# Reordenar columnas: clase al final
nuevo_orden <- c(setdiff(colnames(dataset), PARAMS$feature_engineering$const$clase),
                 PARAMS$feature_engineering$const$clase)
setcolorder(dataset, nuevo_orden)

# Corregir NaNs surgidos de divisiones (ej: en ratios)
cols_lagueables <- copy(setdiff(colnames(dataset),
                                PARAMS$feature_engineering$const$campos_fijos))
for (col in cols_lagueables) {
  dataset[[col]][is.nan(dataset[[col]])] <- 0
}

# Filtrar observaciones: solo mantener filas con clase definida O año presente
# Las filas del año presente NO tienen clase (porque queremos predecir el futuro)
dataset <- dataset[(!is.na(get(PARAMS$feature_engineering$const$clase))) |
                     year >= PARAMS$feature_engineering$const$presente]

# ----------------------------------------------------------------------------
# GUARDAR DATASET PROCESADO
# ----------------------------------------------------------------------------

# Crear estructura de directorios para el experimento
experiment_dir <- paste(PARAMS$experiment$experiment_label,
                        PARAMS$experiment$experiment_code, sep = "_")
experiment_lead_dir <- paste(PARAMS$experiment$experiment_label,
                             PARAMS$experiment$experiment_code,
                             paste0("f", PARAMS$feature_engineering$const$orden_lead),
                             sep = "_")

setwd(PARAMS$environment$base_dir)
setwd(paste0(PARAMS$experiment$exp_dir))

dir.create(experiment_dir, showWarnings = FALSE)
setwd(experiment_dir)
dir.create(experiment_lead_dir, showWarnings = FALSE)
setwd(experiment_lead_dir)

# Guardar metadata del experimento en JSON
PARAMS$features$features_n <- length(colnames(dataset))
PARAMS$features$colnames <- colnames(dataset)

jsontest <- jsonlite::toJSON(PARAMS, pretty = TRUE, auto_unbox = TRUE)
sink(file = paste0(experiment_lead_dir, ".json"))
print(jsontest)
sink(file = NULL)

# Guardar dataset en CSV comprimido
dir.create("01_FE", showWarnings = FALSE)
setwd("01_FE")

fwrite(dataset,
       paste0(experiment_lead_dir, ".csv.gz"),
       logical01 = TRUE,
       sep = ",")

# Mensaje de confirmación
cat("\n=== FEATURE ENGINEERING HEALTH ECONOMICS COMPLETADO ===\n")
cat("Dimensiones finales del dataset:", nrow(dataset), "x", ncol(dataset), "\n")
cat("Archivo guardado exitosamente\n")
cat("\nPróximos pasos:\n")
cat("1. Revisar el archivo JSON generado para ver qué variables se crearon\n")
cat("2. Ejecutar el script 02_TS_health.R para particionar los datos\n")
cat("3. Experimentar creando tus propias variables en AgregarVariables()\n")
cat("\n¡Buena suerte con el desafío!\n")
