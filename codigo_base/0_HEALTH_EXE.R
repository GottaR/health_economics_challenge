#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

library(data.table)
library(lubridate)

#################### Definicion Parametros ######################
library(yaml)
carpeta_base <- "C:/Users/julie/OneDrive/Escritorio/UNO/Aplicaciones/health_economics_challenge"
#carpeta_base <- "C:/Users/rgotta_enterbyte/Desktop/Uno/Aplicaciones/health_economics_challenge"
setwd(carpeta_base)
objetos_trans_script <- c("experiment_dir","experiment_lead_dir","carpeta_base","objetos_trans_script")

PARAMS <- yaml.load_file("./codigo_base/CONFIG_alta_complejidad_full_con_todos_los_ratios.yml")


# Carpetas de experimento
experiment_dir <- paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,sep = "_")
experiment_lead_dir <- paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,paste0("f",PARAMS$feature_engineering$const$orden_lead),sep = "_")

setwd(carpeta_base)
dir.create("exp", showWarnings = FALSE)
setwd("exp")

dir.create(experiment_dir,showWarnings = FALSE)
setwd(experiment_dir)
dir.create(experiment_lead_dir,showWarnings = FALSE)
setwd(experiment_lead_dir)

#################### Redefinir tiempos para health economics ##########################
# Para datos de salud, trabajamos con años en lugar de meses
presente_year <- PARAMS$feature_engineering$const$presente
canaritos_year_end <- presente_year - PARAMS$feature_engineering$const$orden_lead - 1
PARAMS$feature_engineering$const$canaritos_year_end <- canaritos_year_end

canaritos_year_valid <- presente_year - PARAMS$feature_engineering$const$orden_lead
PARAMS$feature_engineering$const$canaritos_year_valid <- canaritos_year_valid
#-------------------------------------------------------------------------

#################################################################
# Persisto los parametros en un json
jsontest = jsonlite::toJSON(PARAMS, pretty = TRUE, auto_unbox = TRUE, null = "null")
write(jsontest,paste0(experiment_lead_dir,".json"))

################### Feature Engineering ################### 
setwd(carpeta_base)
setwd("./codigo_base")
source(PARAMS$feature_engineering$files$fe_script)

################### Training Strategy ###################

#limpio la memoria
rm( list=setdiff(ls(),objetos_trans_script) )  #remove objects
gc()             #garbage collection

setwd(carpeta_base)
setwd("exp")
setwd(experiment_dir)
setwd(experiment_lead_dir)

jsonfile <- list.files(pattern = ".json")
PARAMS <- jsonlite::fromJSON(jsonfile)

setwd(carpeta_base)
setwd("./codigo_base")
source(PARAMS$training_strategy$files$ts_script)

#################################################################
setwd(carpeta_base)
setwd("exp")
setwd(experiment_dir)
setwd(experiment_lead_dir)
# actualizo los parametros del json
jsontest = jsonlite::toJSON(PARAMS, pretty = TRUE, auto_unbox = TRUE, null = "null")
write(jsontest,paste0(experiment_lead_dir,".json"))

################### Hyperparameter Tuning ###################
#limpio la memoria
rm( list=setdiff(ls(),objetos_trans_script) )  #remove objects
gc()             #garbage collection

setwd(carpeta_base)
setwd("exp")
setwd(experiment_dir)
setwd(experiment_lead_dir)

jsonfile <- list.files(pattern = ".json")
PARAMS <- jsonlite::fromJSON(jsonfile)

setwd(carpeta_base)
setwd("./codigo_base")
source(PARAMS$hyperparameter_tuning$files$ht_script)

cat("Pipeline Health Economics completado exitosamente!\n")

