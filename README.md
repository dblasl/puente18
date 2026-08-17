# PUENTE 18+ — MVP

Prototipo web para coordinar la transición de adolescentes con condiciones crónicas, raras o complejas hacia servicios de adultos.

## Qué incluye

- Dashboard de gestión con KPIs y semáforos.
- Ficha de transición 360° con datos clínicos sintéticos.
- Navegador de derivación por especialidad, complejidad, territorio y capacidad reportada.
- Vista de capacidad de red con heatmap.
- Módulo inicial de autoaprendizaje para joven/tutor.
- Motor de riesgo basado en reglas.
- Módulo de demostración ML con datos sintéticos.
- Carga de Excel/CSV.
- Script ejecutable en Google Colab.
- Arquitectura preparada para sustituir las fuentes sintéticas por APIs/fuentes institucionales en una fase posterior.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar para la demo

Para una demo de hackathon, la ruta recomendada es GitHub -> Streamlit Community Cloud. No usar historias clínicas reales ni datos identificables.

## Datos

Todo lo incluido en `data/` es sintético y se usa únicamente para demostración.

## Modelo predictivo

La versión del MVP usa un motor de reglas como capa explicable y un pequeño modelo de regresión logística entrenado sobre datos sintéticos para demostrar el pipeline técnico. Su AUC y predicciones no son evidencia clínica.

## Evolución

1. Validar variables y outcome con profesionales.
2. Reemplazar datos sintéticos por datos anonimizados y gobernados.
3. Validar externalmente el modelo.
4. Integrar fuentes institucionales/APIs.
5. Evolucionar hacia estándares de interoperabilidad (p. ej. HL7 FHIR/IPS) cuando corresponda.
