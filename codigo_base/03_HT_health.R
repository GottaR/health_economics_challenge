# ============================================================================
# HYPERPARAMETER TUNING v2 - CORREGIDO
# Universidad del Oeste (UNO) - Aplicaciones en Ciencia de Datos
# ============================================================================
# DIFERENCIA CON v1:
# - v1: El RMSE para la BO venía del conjunto VALIDATE (mismo que early stopping)
# - v2: El RMSE para la BO viene del conjunto TEST (evaluación independiente)
#
# Esto es metodológicamente más correcto porque:
# - VALIDATE se usa SOLO para early stopping (cuándo parar de entrenar)
# - TEST se usa SOLO para evaluar la calidad de los hiperparámetros
# - Son dos propósitos distintos, deben ser conjuntos distintos
# ============================================================================

# ----------------------------------------------------------------------------
# LIBRERÍAS NECESARIAS
# ----------------------------------------------------------------------------
require("data.table")
require("primes")
require("lightgbm")
require("DiceKriging")
require("mlrMBO")
require("rlist")

setDTthreads(percent = 65)

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

loguear <- function(reg, arch = NA, folder = "./exp/", ext = ".txt", verbose = TRUE) {
  archivo <- arch
  if (is.na(arch)) archivo <- paste0(folder, substitute(reg), ext)

  if (!file.exists(archivo)) {
    linea <- paste0("fecha\t",
                    paste(list.names(reg), collapse = "\t"), "\n")
    cat(linea, file = archivo)
  }

  linea <- paste0(format(Sys.time(), "%Y%m%d %H%M%S"), "\t",
                  gsub(", ", "\t", toString(reg)), "\n")

  cat(linea, file = archivo, append = TRUE)
  if (verbose) cat(linea)
}

parametrizar <- function(lparam) {
  param_fijos <- copy(lparam)
  hs <- list()

  for (param in names(lparam)) {
    if (length(lparam[[param]]) > 1) {
      desde <- as.numeric(lparam[[param]][[1]])
      hasta <- as.numeric(lparam[[param]][[2]])

      if (length(lparam[[param]]) == 2) {
        hs <- append(hs,
                     list(makeNumericParam(param, lower = desde, upper = hasta)))
      } else {
        hs <- append(hs,
                     list(makeIntegerParam(param, lower = desde, upper = hasta)))
      }
      param_fijos[[param]] <- NULL
    }
  }

  return(list("param_fijos" = param_fijos,
              "paramSet" = hs))
}

particionar <- function(data, division, agrupa = "", campo = "fold", start = 1, seed = NA) {
  if (!is.na(seed)) set.seed(seed)

  bloque <- unlist(mapply(function(x, y) { rep(y, x) },
                         division,
                         seq(from = start, length.out = length(division))))

  data[, (campo) := sample(rep(bloque, ceiling(.N / length(bloque))))[1:.N],
       by = agrupa]
}

# ============================================================================
# FUNCIÓN PRINCIPAL: ESTIMAR ERROR EN TEST (CORREGIDA)
# ============================================================================
# CAMBIO PRINCIPAL:
# Antes: retornaba record_evals$valid (RMSE de validate)
# Ahora: retorna RMSE calculado sobre TEST
# ============================================================================

EstimarGanancia_lightgbm <- function(x) {
  gc()

  GLOBAL_iteracion <<- GLOBAL_iteracion + 1

  param_completo <- c(param_fijos, x)

  param_completo$num_iterations <- ifelse(param_fijos$boosting == "dart", 999, 99999)
  param_completo$early_stopping_rounds <- as.integer(200 + 4 / param_completo$learning_rate)

  # Entrenar modelo con TRAIN, early stopping con VALIDATE
  set.seed(param_completo$seed)
  modelo_train <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalidate),
    param = param_completo,
    verbose = -100
  )

  # ----------------------------------------------------------------------------
  # EVALUAR EL MODELO EN TEST (CORREGIDO)
  # ----------------------------------------------------------------------------
  # Ahora SÍ usamos las predicciones sobre TEST para calcular el RMSE
  
  prediccion <- predict(modelo_train,
                       data.matrix(dataset_test[, campos_buenos, with = FALSE]))

  # Obtener valores reales del TEST
  valores_reales <- dataset_test[[PARAMS$hyperparameter_tuning$const$campo_clase]]
  
  # ============================================================================
  # CAMBIO CLAVE: Calcular RMSE sobre TEST, no sobre VALIDATE
  # ============================================================================
  rmse_test <- sqrt(mean((prediccion - valores_reales)^2, na.rm = TRUE))
  
  # Válvula de escape: si hay NaN, retornar un valor muy alto
  if (is.nan(rmse_test) || is.na(rmse_test)) {
    rmse_test <- Inf
  }

  gc()

  # ----------------------------------------------------------------------------
  # GUARDAR IMPORTANCIA DE VARIABLES SI ES EL MEJOR MODELO HASTA AHORA
  # ----------------------------------------------------------------------------
  if (rmse_test < GLOBAL_ganancia) {
    GLOBAL_ganancia <<- rmse_test
    tb_importancia <- as.data.table(lgb.importance(modelo_train))

    fwrite(tb_importancia,
           file = paste0(PARAMS$hyperparameter_tuning$files$output$importancia,
                        GLOBAL_iteracion, ".txt"),
           sep = "\t")
  }

  # ----------------------------------------------------------------------------
  # LOGUEAR RESULTADOS
  # ----------------------------------------------------------------------------
  # Agregamos tanto el RMSE de validate (para referencia) como el de test (usado para BO)
  
  rmse_validate <- unlist(modelo_train$record_evals$valid[[PARAMS$hyperparameter_tuning$param$lightgbm$metric]]$eval)[modelo_train$best_iter]

  ds <- list("cols" = ncol(dtrain), "rows" = nrow(dtrain))
  xx <- c(ds, copy(param_completo))

  xx$early_stopping_rounds <- NULL
  xx$num_iterations <- modelo_train$best_iter
  xx$rmse_validate <- rmse_validate  # Para comparar
  xx$rmse_test <- rmse_test          # Este es el que usa la BO
  xx$ganancia <- rmse_test           # Mantener compatibilidad con nombre anterior
  xx$iteracion_bayesiana <- GLOBAL_iteracion

  loguear(xx, arch = PARAMS$hyperparameter_tuning$files$output$BOlog)

  # Retornar RMSE de TEST a la Optimización Bayesiana
  return(rmse_test)
}

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

cat("\n=== INICIANDO HYPERPARAMETER TUNING v2 (CORREGIDO) ===\n")
cat("Diferencia: El RMSE para la BO ahora viene de TEST, no de VALIDATE\n\n")

set.seed(PARAMS$hyperparameter_tuning$param$semilla)

setwd(paste0(carpeta_base, "/exp"))
setwd(experiment_dir)
setwd(experiment_lead_dir)
setwd("02_TS")

# ----------------------------------------------------------------------------
# PASO 1: CARGAR DATASET
# ----------------------------------------------------------------------------
cat("Paso 1: Cargando dataset con training strategy...\n")
nom_arch <- PARAMS$hyperparameter_tuning$files$input$dentrada
dataset <- fread(nom_arch)
cat("Dataset cargado:", nrow(dataset), "filas x", ncol(dataset), "columnas\n\n")

setwd(paste0(carpeta_base, "/exp"))
setwd(experiment_dir)
setwd(experiment_lead_dir)
dir.create("03_HT", showWarnings = FALSE)
setwd("03_HT")

# ----------------------------------------------------------------------------
# PASO 2: PREPARAR DATASETS PARA LIGHTGBM
# ----------------------------------------------------------------------------
cat("Paso 2: Preparando datasets para LightGBM...\n")

campos_buenos <- setdiff(copy(colnames(dataset)),
                        c(PARAMS$hyperparameter_tuning$const$campo_clase,
                          "part_train", "part_validate", "part_test"))

cat("Variables predictoras:", length(campos_buenos), "\n")

# TRAIN
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[part_train == 1, campos_buenos, with = FALSE]),
  label = dataset[part_train == 1][[PARAMS$hyperparameter_tuning$const$campo_clase]],
  free_raw_data = FALSE
)
cat("Train:", nrow(dataset[part_train == 1]), "registros\n")

# VALIDATE (solo para early stopping)
dvalidate <- lgb.Dataset(
  data = data.matrix(dataset[part_validate == 1, campos_buenos, with = FALSE]),
  label = dataset[part_validate == 1][[PARAMS$hyperparameter_tuning$const$campo_clase]],
  free_raw_data = FALSE
)
cat("Validate:", nrow(dataset[part_validate == 1]), "registros (para early stopping)\n")

# TEST (para evaluar HP - CAMBIO CLAVE)
dataset_test <- dataset[part_test == 1]
cat("Test:", nrow(dataset_test), "registros (para evaluar HP)\n\n")

rm(dataset)
gc()

# ----------------------------------------------------------------------------
# PASO 3: PREPARAR OPTIMIZACIÓN BAYESIANA
# ----------------------------------------------------------------------------
cat("Paso 3: Configurando Optimización Bayesiana...\n")

hiperparametros <- PARAMS$hyperparameter_tuning$param[[PARAMS$hyperparameter_tuning$param$algoritmo]]
apertura <- parametrizar(hiperparametros)
param_fijos <- apertura$param_fijos

cat("Hiperparámetros fijos:", length(param_fijos), "\n")
cat("Hiperparámetros a optimizar:", length(apertura$paramSet), "\n")

cat("\nHiperparámetros a optimizar:\n")
for (i in seq_along(apertura$paramSet)) {
  param_name <- apertura$paramSet[[i]]$id
  param_lower <- apertura$paramSet[[i]]$lower
  param_upper <- apertura$paramSet[[i]]$upper
  cat("  -", param_name, ": [", param_lower, ",", param_upper, "]\n")
}
cat("\n")

# Inicializar variables globales
if (file.exists(PARAMS$hyperparameter_tuning$files$output$BOlog)) {
  cat("Detectado log previo. Retomando desde iteración anterior...\n")
  tabla_log <- fread(PARAMS$hyperparameter_tuning$files$output$BOlog)
  GLOBAL_iteracion <- nrow(tabla_log)
  GLOBAL_ganancia <- tabla_log[, min(ganancia)]
  cat("Iteración inicial:", GLOBAL_iteracion, "\n")
  cat("Mejor RMSE (test) hasta ahora:", GLOBAL_ganancia, "\n\n")
  rm(tabla_log)
} else {
  GLOBAL_iteracion <- 0
  GLOBAL_ganancia <- Inf
  cat("Comenzando optimización desde cero.\n\n")
}

# ----------------------------------------------------------------------------
# PASO 4: EJECUTAR OPTIMIZACIÓN BAYESIANA
# ----------------------------------------------------------------------------
cat("Paso 4: Configurando mlrMBO...\n")

funcion_optimizar <- EstimarGanancia_lightgbm

configureMlr(show.learner.output = FALSE)

obj.fun <- makeSingleObjectiveFunction(
  fn = funcion_optimizar,
  minimize = PARAMS$hyperparameter_tuning$param$BO$minimize,
  noisy = PARAMS$hyperparameter_tuning$param$BO$noisy,
  par.set = makeParamSet(params = apertura$paramSet),
  has.simple.signature = PARAMS$hyperparameter_tuning$param$BO$has.simple.signature
)

ctrl <- makeMBOControl(
  save.on.disk.at.time = PARAMS$hyperparameter_tuning$param$BO$save.on.disk.at.time,
  save.file.path = PARAMS$hyperparameter_tuning$files$output$BObin
)

ctrl <- setMBOControlTermination(
  ctrl,
  iters = PARAMS$hyperparameter_tuning$param$BO$iterations
)

ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

surr.km <- makeLearner(
  "regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  control = list(trace = TRUE)
)

cat("Configuración completa.\n")
cat("Iteraciones a ejecutar:", PARAMS$hyperparameter_tuning$param$BO$iterations, "\n")
cat("\n=== INICIANDO OPTIMIZACIÓN BAYESIANA ===\n")
cat("NOTA: El RMSE se calcula sobre TEST (no VALIDATE)\n")
cat("El progreso se guarda en:", PARAMS$hyperparameter_tuning$files$output$BOlog, "\n\n")

if (!file.exists(PARAMS$hyperparameter_tuning$files$output$BObin)) {
  run <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else {
  cat("Retomando optimización desde archivo guardado...\n")
  run <- mboContinue(PARAMS$hyperparameter_tuning$files$output$BObin)
}

cat("\n=== OPTIMIZACIÓN BAYESIANA COMPLETADA ===\n\n")

# ----------------------------------------------------------------------------
# PASO 5: ENTRENAR MODELO FINAL
# ----------------------------------------------------------------------------
cat("Paso 5: Entrenando modelo final con mejores hiperparámetros...\n")

if (file.exists(PARAMS$hyperparameter_tuning$files$output$BOlog)) {

  tabla_log <- fread(PARAMS$hyperparameter_tuning$files$output$BOlog)
  mejor_iteracion <- tabla_log[which.min(ganancia)]

  cat("\n=== MEJORES HIPERPARÁMETROS ENCONTRADOS ===\n")
  cat("RMSE en TEST:", mejor_iteracion$ganancia, "\n")
  if ("rmse_validate" %in% names(mejor_iteracion)) {
    cat("RMSE en VALIDATE:", mejor_iteracion$rmse_validate, "\n")
  }
  cat("Iteración:", mejor_iteracion$iteracion_bayesiana, "\n")
  cat("Iteraciones de boosting:", mejor_iteracion$num_iterations, "\n\n")

  mejores_params <- param_fijos

  for (col in names(apertura$paramSet)) {
    if (col %in% names(mejor_iteracion)) {
      mejores_params[[col]] <- mejor_iteracion[[col]]
      cat(col, "=", mejor_iteracion[[col]], "\n")
    }
  }

  mejores_params$num_iterations <- mejor_iteracion$num_iterations
  mejores_params$early_stopping_rounds <- NULL

  cat("\n")

  # ENTRENAR CON TRAIN_FINAL
  cat("Cargando train_final (todos los datos históricos)...\n")
  dataset_final <- fread(paste0("../02_TS/", PARAMS$training_strategy$files$output$train_final))

  cat("Train_final:", nrow(dataset_final), "filas x", ncol(dataset_final), "columnas\n")

  campos_buenos_final <- setdiff(copy(colnames(dataset_final)),
                                c(PARAMS$hyperparameter_tuning$const$campo_clase))

  dtrain_final <- lgb.Dataset(
    data = data.matrix(dataset_final[, campos_buenos_final, with = FALSE]),
    label = dataset_final[[PARAMS$hyperparameter_tuning$const$campo_clase]],
    free_raw_data = FALSE
  )

  cat("Entrenando modelo final...\n")
  set.seed(mejores_params$seed)
  modelo_final <- lgb.train(
    data = dtrain_final,
    param = mejores_params,
    verbose = 100
  )

  cat("\nModelo final entrenado exitosamente.\n")

  # Guardar modelo
  saveRDS(modelo_final, file = "modelo_final_lgb.rds")
  cat("Modelo guardado como: modelo_final_lgb.rds\n")

  # Guardar importancia
  tb_importancia_final <- as.data.table(lgb.importance(modelo_final))
  fwrite(tb_importancia_final,
         file = PARAMS$hyperparameter_tuning$files$output$tb_importancia,
         sep = "\t")
  cat("Importancia de variables guardada como:",
      PARAMS$hyperparameter_tuning$files$output$tb_importancia, "\n")

  cat("\n=== TOP 10 VARIABLES MÁS IMPORTANTES ===\n")
  print(tb_importancia_final[1:min(10, nrow(tb_importancia_final))])
  cat("\n")

  # ----------------------------------------------------------------------------
  # PASO 6: GENERAR PREDICCIONES PARA 2022
  # ----------------------------------------------------------------------------
  cat("Paso 6: Generando predicciones para el año 2022...\n")

  if (file.exists(paste0("../02_TS/", PARAMS$training_strategy$files$output$present_data))) {

    dataset_present <- fread(paste0("../02_TS/", PARAMS$training_strategy$files$output$present_data))

    if (nrow(dataset_present) > 0) {

      cat("Datos presentes (año 2021):", nrow(dataset_present), "registros\n")

      campos_present <- intersect(campos_buenos_final, names(dataset_present))

      cat("Aplicando modelo...\n")
      predicciones_present <- predict(
        modelo_final,
        data.matrix(dataset_present[, campos_present, with = FALSE])
      )

      dataset_present[, prediccion_clase := predicciones_present]

      fwrite(dataset_present, file = "predicciones_presente.csv")

      cat("Predicciones guardadas como: predicciones_presente.csv\n")
      cat("\nEstadísticas de las predicciones:\n")
      cat("  - Media:", mean(predicciones_present), "\n")
      cat("  - Mediana:", median(predicciones_present), "\n")
      cat("  - Mínimo:", min(predicciones_present), "\n")
      cat("  - Máximo:", max(predicciones_present), "\n")
      cat("  - Desv. Est.:", sd(predicciones_present), "\n")

    } else {
      cat("No hay datos presentes para predecir.\n")
    }

  } else {
    cat("Archivo de datos presentes no encontrado.\n")
  }

  rm(tabla_log)
}

# ----------------------------------------------------------------------------
# RESUMEN FINAL
# ----------------------------------------------------------------------------
cat("\n=== HYPERPARAMETER TUNING v2 COMPLETADO ===\n\n")

cat("DIFERENCIA CON v1:\n")
cat("  - v1: RMSE para BO venía de VALIDATE (mismo conjunto que early stopping)\n")
cat("  - v2: RMSE para BO viene de TEST (evaluación independiente)\n\n")

cat("ARCHIVOS GENERADOS:\n")
cat("  1.", PARAMS$hyperparameter_tuning$files$output$BOlog, "\n")
cat("     (ahora incluye rmse_validate Y rmse_test para comparar)\n")
cat("  2. modelo_final_lgb.rds\n")
cat("  3.", PARAMS$hyperparameter_tuning$files$output$tb_importancia, "\n")
cat("  4. predicciones_presente.csv\n\n")
