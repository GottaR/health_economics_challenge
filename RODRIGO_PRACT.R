install.packages(c(
  "data.table",
  "lightgbm",
  "yaml",
  "mlrMBO",
  "ggplot2",
  "dplyr",
  "stringr",
  "lubridate"))

library(zoo)      
install.packages("randomForest")
library(randomForest)  
install.packages("data.table")
library(data.table)
setDT(dataset)

library(readr)
dataset_desafio <- read_csv("dataset/dataset_desafio.csv")
cat("Filas leídas:", nrow(dataset_desafio), " columnas:", ncol(dataset_desafio), "\n")

dataset <- as.data.table(dataset_desafio)
names(dataset)

if (!("Country Code" %in% names(dataset))) stop("Falta columna 'Country Code' en dataset.")
if (!("year" %in% names(dataset))) stop("Falta columna 'year' en dataset.")
setkey(dataset, `Country Code`, year)

if (!dir.exists("dataset")) dir.create("dataset")
if (!dir.exists("exp")) dir.create("exp")

### VARIABLE EFICIENCIA 

setDT(dataset)

setorder(dataset, `Country Name`, year)

# Mortalidad sobre PBI per cápita
dataset[, mort_gdp_pc := SH.DYN.MORT / NY.GDP.PCAP.CD]

# Esperanza de vida sobre PBI per cápita
dataset[, lifeexp_gdp_pc := SP.DYN.LE00.IN / NY.GDP.PCAP.CD]

# Mortalidad neonatal sobre PBI per cápita
dataset[, nmort_gdp_pc := SH.DYN.NMRT / NY.GDP.PCAP.CD]

# Eficiencia inversa: años de vida por dólar del PBI per cápita
dataset[, efficiency_life_dollar := SP.DYN.LE00.IN / NY.GDP.PCAP.CD]

cat("\n---- mort_gdp_pc ----\n")
print(head(dataset$mort_gdp_pc, 10))
print(summary(dataset$mort_gdp_pc))

cat("\n---- lifeexp_gdp_pc ----\n")
print(head(dataset$lifeexp_gdp_pc, 10))
print(summary(dataset$lifeexp_gdp_pc))

cat("\n---- nmort_gdp_pc ----\n")
print(head(dataset$nmort_gdp_pc, 10))
print(summary(dataset$nmort_gdp_pc))

cat("\n---- efficiency_life_dollar ----\n")
print(head(dataset$efficiency_life_dollar, 10))
print(summary(dataset$efficiency_life_dollar))


### VARIABLES TENDENCIA 

setorderv(dataset, c("Country Code", "year"))

# life_exp_change: diferencia año a año por país
dataset[, life_exp_change := NA_real_]
dataset[, life_exp_change := c(NA, diff(SP.DYN.LE00.IN)), by = `Country Code`]

cat("VAR TENDENCIA: resumen life_exp_change:\n"); print(summary(dataset$life_exp_change))
cat("VAR TENDENCIA: resumen gdp_growth_volatility:\n"); print(summary(dataset$gdp_growth_volatility))

# Tendencia de esperanza de vida
dataset[, trend_lifeexp := c(NA, diff(SP.DYN.LE00.IN)), by = `Country Name`]

# Tendencia mortalidad general
dataset[, trend_mort := c(NA, diff(SH.DYN.MORT)), by = `Country Name`]

# Tendencia mortalidad neonatal
dataset[, trend_nmort := c(NA, diff(SH.DYN.NMRT)), by = `Country Name`]

# Tendencia del PBI per cápita
dataset[, trend_gdp_pc := c(NA, diff(NY.GDP.PCAP.CD)), by = `Country Name`]

cat("\n---- trend_lifeexp ----\n")
print(head(dataset$trend_lifeexp, 10))
print(summary(dataset$trend_lifeexp))

cat("\n---- trend_mort ----\n")
print(head(dataset$trend_mort, 10))
print(summary(dataset$trend_mort))

cat("\n---- trend_nmort ----\n")
print(head(dataset$trend_nmort, 10))
print(summary(dataset$trend_nmort))

cat("\n---- trend_gdp_pc ----\n")
print(head(dataset$trend_gdp_pc, 10))
print(summary(dataset$trend_gdp_pc))


### VARIABLE CONTEXTO 

# crisis 2008-2009
dataset[, crisis_2008 := ifelse(year %in% 2008:2009, 1L, 0L)]

# high_income 
if ("income" %in% names(dataset)) {
  dataset[, high_income := ifelse(income == "High", 1L, 0L)]} else {
  dataset[, high_income := NA_integer_]
  warning("No se encontró columna 'income' — high_income cargado como NA.")}

cat("VAR CONTEXTO: crisis_2008 distribution:\n"); print(table(dataset$crisis_2008, useNA = "ifany"))
if ("high_income" %in% names(dataset)) cat("VAR CONTEXTO: high_income NAs:", sum(is.na(dataset$high_income)), "\n")

# Ratio población urbana
dataset[, urban_ratio := SP.URB.TOTL / SP.POP.TOTL]

# Ratio dependencia poblacional
dataset[, dependency_ratio := SP.POP.DPND / SP.POP.TOTL]

# Crecimiento poblacional
dataset[, pop_growth := SP.POP.GROW]

cat("\n---- urban_ratio ----\n")
print(head(dataset$urban_ratio, 10))
print(summary(dataset$urban_ratio))

cat("\n---- dependency_ratio ----\n")
print(head(dataset$dependency_ratio, 10))
print(summary(dataset$dependency_ratio))

cat("\n---- pop_growth ----\n")
print(head(dataset$pop_growth, 10))
print(summary(dataset$pop_growth))

cat("\n---- health_ratio_example ----\n")
print(head(dataset$health_ratio_example, 10))
print(summary(dataset$health_ratio_example))

cat("\n---- log_gdp_pc ----\n")
print(head(dataset$log_gdp_pc, 10))
print(summary(dataset$log_gdp_pc))

cat("\n---- log_pop ----\n")
print(head(dataset$log_pop, 10))
print(summary(dataset$log_pop))

## RADOM FOREST
install.packages("randomForest")
library(randomForest)
library(data.table)
library(ggplot2)

setDT(dataset)


vars_modelo <- c(
  "efficiency_life_dollar",   # variable objetivo
  "mort_gdp_pc", "lifeexp_gdp_pc", "nmort_gdp_pc",
  "trend_lifeexp", "trend_mort", "trend_nmort", "trend_gdp_pc",
  "urban_ratio", "dependency_ratio", "pop_growth",
  "crisis_2008", "high_income")

vars_modelo <- vars_modelo[vars_modelo %in% names(dataset)]

dataset_rf <- dataset[, ..vars_modelo]

##Eliminamos filas con NA

dataset_rf <- na.omit(dataset_rf)

cat("Filas antes del modelo:", nrow(dataset), "\n")
cat("Filas después de limpiar NAs:", nrow(dataset_rf), "\n")

set.seed(123)

modelo_rf <- randomForest(
  efficiency_life_dollar ~ .,
  data = dataset_rf,
  ntree = 500,
  mtry = 4,
  importance = TRUE)

print(modelo_rf)


importancia <- importance(modelo_rf)
print(importancia)

# Convertimos para graficar
imp_df <- data.frame(
  Variable = rownames(importancia),
  IncMSE = importancia[, "%IncMSE"])


ggplot(imp_df, aes(x = reorder(Variable, IncMSE), y = IncMSE)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(
    title = "Importancia de Variables - Random Forest",
    x = "Variable",
    y = "% Aumento del MSE al remover la variable"
  ) +
  theme_minimal()

## PREDICCIONES 

dataset_rf$pred_rf <- predict(modelo_rf, newdata = dataset_rf)

head(dataset_rf[, .(efficiency_life_dollar, pred_rf)])
