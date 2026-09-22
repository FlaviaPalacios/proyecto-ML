# Proyecto Final — Machine Learning
## Grupo 11

## 1. Título del proyecto
Predicción de riesgo de default en préstamos personales: un enfoque honesto sin leakage sobre Lending Club Loan Data

## 2. Integrantes
- Shillta Huaman, Dalesska Belinda Isabel 202320097
- Palacios Dávalos, Flavia Luciana 202110258
- Joseph Jossemy, Cabanillas Solis 202410347


## 3. Dataset elegido
**Lending Club Loan Data** (Kaggle / Lending Club)
URL: https://www.kaggle.com/datasets/wordsforthewise/lending-club

Dataset de préstamos personales originados por la plataforma Lending Club (EE.UU.), con información de la solicitud, características del solicitante y desempeño del préstamo a lo largo del tiempo. Cumple los criterios de complejidad exigidos: más de 2 millones de filas, más de 140 variables, fuerte componente temporal (vintages de originación), variables de texto libre, alto porcentaje de valores faltantes en varias columnas y fuerte desbalance de clases en la variable objetivo.

**Licencia / acceso:** dataset público en Kaggle, sin restricciones de acceso especiales (a diferencia de MIMIC-IV). No requiere acuerdos de uso de datos.

## 4. Pregunta predictiva
¿Un préstamo terminará en incumplimiento (default / charge-off), usando únicamente la información disponible en el momento en que Lending Club decide originar el préstamo?

Este es un problema de **estimación de riesgo crediticio**, enmarcado como clasificación binaria. Como extensión (si el tiempo lo permite), se comparará el desempeño de un modelo interpretable (regresión logística) contra un modelo de alto rendimiento (gradient boosting), discutiendo el trade-off interpretabilidad vs. desempeño — uno de los problemas sugeridos explícitamente para este dataset.

## 5. Variable objetivo
Variable binaria derivada de `loan_status`:
- **Clase positiva / "malo" (1):** `Charged Off`, `Default`
- **Clase negativa / "bueno" (0):** `Fully Paid`

Los préstamos con estado `Current`, `In Grace Period`, `Late (16-30 days)` o `Late (31-120 days)` se **excluyen del conjunto de modelado**, ya que su desenlace final aún no está determinado; incluirlos como "buenos" introduciría sesgo de censura (censoring bias).

## 6. Unidad de predicción
Un préstamo individual (`id`), evaluado en el instante de la decisión de originación (antes del desembolso de fondos).

## 7. Variables disponibles antes de la predicción
**Incluidas (conocidas al momento de la solicitud):**
`loan_amnt`, `term`, `purpose`, `annual_inc`, `dti`, `emp_length`, `emp_title`, `home_ownership`, `verification_status`, `fico_range_low`, `fico_range_high`, `open_acc`, `total_acc`, `revol_bal`, `revol_util`, `delinq_2yrs`, `earliest_cr_line`, `pub_rec`, `inq_last_6mths`, `mort_acc`, `application_type`, `addr_state`, `issue_d` (usada solo para el split temporal, no como feature).

**Zona gris — se probará con y sin estas variables:**
`grade`, `sub_grade`, `int_rate`. Son asignadas por el propio modelo de riesgo de Lending Club en el momento de originación, por lo que técnicamente están disponibles, pero un modelo que las use está parcialmente replicando el scoring de LC en vez de aprender riesgo desde cero. Se documentará el efecto de incluirlas o no.

**Excluidas por leakage evidente (información posterior al desembolso):**
`total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`, `out_prncp`, `out_prncp_inv`, `last_credit_pull_d`, `hardship_*`, `settlement_*`, `debt_settlement_flag`.

## 8. Riesgos de leakage
- **Leakage temporal directo:** variables de pago/recuperación (ver lista de excluidas) están perfectamente correlacionadas con el desenlace porque se generan después de que el préstamo ya incumplió o se pagó. Control: exclusión explícita antes de cualquier entrenamiento.
- **Leakage vía variables "zona gris":** `grade`/`sub_grade`/`int_rate` podrían filtrar información del propio proceso de decisión de LC. Control: entrenar dos versiones del modelo (con y sin estas variables) y comparar.
- **Leakage temporal en el split:** si se hace un split aleatorio en vez de temporal, el modelo podría "ver" información macroeconómica de un período (ej. una recesión) que también aparece en el set de test. Control: split por `issue_d` (out-of-time validation), no aleatorio.
- **Censura de la variable objetivo:** incluir préstamos aún activos (`Current`) como "buenos" sesgaría el modelo. Control: excluir estados no definitivos.

## 9. Métrica principal y métrica secundaria
- **Métrica principal:** AUC-PR (área bajo la curva Precision-Recall). Se prioriza sobre accuracy y sobre AUC-ROC porque la clase de interés (default) es minoritaria (~15-20% del dataset filtrado), y AUC-PR es más sensible al desempeño sobre la clase positiva bajo desbalance.
- **Métrica secundaria:** Recall a un umbral de precisión fijo (ej. precision ≥ 0.4), y KS statistic (Kolmogorov-Smirnov), estándar en la industria de riesgo crediticio para medir separación entre distribuciones de buenos y malos.
- Se reportará también la matriz de confusión y el classification report completo — nunca solo accuracy, dado el desbalance.

## 10. Plan de validación
**Validación fuera de tiempo (out-of-time), no aleatoria**, dado el fuerte componente temporal del dataset:
- Entrenamiento: préstamos originados en 2015–2017
- Validación (tuning de hiperparámetros): préstamos originados en 2018
- Test final (una sola evaluación, al final): préstamos originados en 2019

Dentro del set de entrenamiento se usará validación cruzada estratificada (k=5) únicamente para comparar modelos y ajustar hiperparámetros, nunca para la evaluación final reportada.

## 11. Modelo baseline
Regresión logística con las variables "incluidas" de la sección 7 (sin `grade`/`sub_grade`/`int_rate`), con:
- Imputación simple de faltantes (mediana/moda)
- One-hot encoding de categóricas
- Estandarización de numéricas
- `class_weight="balanced"` para mitigar el desbalance
- Semilla fija (`random_state=42`)

Este baseline honesto sirve como piso de comparación para los modelos posteriores (árboles, gradient boosting).

## 12. Riesgos técnicos
- **Volumen de datos:** el dataset completo supera 2M de filas; puede requerir muestreo estratificado por período o uso de Dask/chunks para exploración inicial.
- **Alta cardinalidad en categóricas:** `emp_title`, `addr_state`, `purpose` requieren tratamiento cuidadoso (agrupar categorías raras, target encoding con precaución de leakage).
- **Desbalance de clases:** puede requerir técnicas adicionales (class weights, undersampling, o ajuste de umbral) más allá del baseline.
- **Drift temporal:** el comportamiento crediticio cambia entre 2015 y 2019 (ciclo económico); el modelo debe evaluarse explícitamente en su capacidad de generalizar a períodos futuros no vistos.
- **Valores faltantes masivos** en columnas específicas (ej. variables de hardship, algunas de co-borrower) que podrían no aportar señal y deben evaluarse para descarte.

## 13. Plan de trabajo semanas restantes
- **Semana 1 (entrega previa):** exploración inicial, definición de variable objetivo, baseline simple.
- **Semanas 2-3:** limpieza y feature engineering, pipeline reproducible en `src/`, tratamiento de faltantes y categóricas.
- **Semanas 4-6:** entrenamiento de modelos adicionales (random forest, gradient boosting), búsqueda de hiperparámetros, comparación con y sin variables "zona gris".
- **Semanas 7-8:** análisis de errores por segmentos, interpretabilidad (SHAP / coeficientes), discusión de sesgos y limitaciones.
- **Última semana:** informe final, presentación, limpieza de código y verificación de reproducibilidad end-to-end.
