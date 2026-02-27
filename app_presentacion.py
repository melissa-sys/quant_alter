import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuración de página
st.set_page_config(
    page_title="Punto 3 - Análisis & Rebalanceo de Portafolios",
    page_icon="📊",
    layout="wide"
)

# Datos de entrada
limites_perfiles = {
    'Conservador': {
        'RF':         {'min': 0.60, 'max': 0.80},
        'FIC':        {'min': 0.15, 'max': 0.30},
        'RV_ALT':     {'min': 0.00, 'max': 0.15}  
    },
    'Moderado': {
        'RF':         {'min': 0.35, 'max': 0.55},
        'FIC':        {'min': 0.20, 'max': 0.35},
        'RV_ALT':     {'min': 0.20, 'max': 0.40} 
    },
    'Arriesgado': {
        'RF':         {'min': 0.05, 'max': 0.25},
        'FIC':        {'min': 0.15, 'max': 0.30},
        'RV_ALT':     {'min': 0.50, 'max': 0.75} 
    }
}

def preparar_datos_portafolio(df_portafolio, limites_perfiles):
    """
    Prepara todos los datos necesarios para el análisis de portafolios.
    
    Esta función encapsula:
    1. Agregación de participaciones por macroactivo y cliente (estado_clientes_actual)
    2. Validación de cumplimiento de límites por perfil de riesgo
    3. Ranking de activos preferidos por calidad (puntaje y Sharpe)
    
    Args:
        df_portafolio: DataFrame con datos de portafolios (portafolios.csv)
        limites_perfiles: dict con límites {'RF': {'min': x, 'max': y}, ...}
    
    Returns:
        tuple: (estado_clientes_actual, activos_preferidos)
            - estado_clientes_actual: DataFrame con participaciones agregadas y cumplimiento
            - activos_preferidos: DataFrame con activos rankeados por calidad
    """
       
    # Agrupar por cliente y macroactivo
    participacion_por_macro = df_portafolio.groupby(
        ['cliente_id', 'macroactivo']
    )['participacion_actual'].sum().reset_index()
    
    # Pivotar para tener columna por macroactivo
    pivot_macro = participacion_por_macro.pivot(
        index='cliente_id',
        columns='macroactivo',
        values='participacion_actual'
    ).fillna(0)
    
    pivot_macro.columns.name = None
    pivot_macro.reset_index(inplace=True)
    
    # Crear columna RV_ALT = Renta Variable + Alternativo
    pivot_macro['RV_ALT'] = (
        pivot_macro.get('Renta Variable', 0) + 
        pivot_macro.get('Alternativo', 0)
    )
    
    # Renombrar columnas
    rename_dict = {'Renta Fija': 'RF', 'Renta Variable': 'RV'}
    pivot_macro.rename(columns=rename_dict, inplace=True)
    
    # Merge con perfil de riesgo
    perfil_clientes = df_portafolio[
        ['cliente_id', 'perfil_riesgo_cliente']
    ].drop_duplicates()
    
    estado_clientes_actual = pivot_macro.merge(
        perfil_clientes, on='cliente_id', how='left'
    )
       
    def validar_limites(row):
        """Valida cumplimiento de límites por perfil"""
        perfil = row['perfil_riesgo_cliente']
        limites = limites_perfiles[perfil]
        
        rf_ok = limites['RF']['min'] <= row['RF'] <= limites['RF']['max']
        fic_ok = limites['FIC']['min'] <= row['FIC'] <= limites['FIC']['max']
        rv_alt_ok = limites['RV_ALT']['min'] <= row['RV_ALT'] <= limites['RV_ALT']['max']
        
        return pd.Series({
            'RF_cumple': rf_ok,
            'FIC_cumple': fic_ok,
            'RV_ALT_cumple': rv_alt_ok,
            'cumple_perfil': rf_ok and fic_ok and rv_alt_ok
        })
    
    cumplimiento = estado_clientes_actual.apply(validar_limites, axis=1)
    estado_clientes_actual = pd.concat([estado_clientes_actual, cumplimiento], axis=1)
    
    # Reordenar columnas
    cols_orden = [
        'cliente_id', 'perfil_riesgo_cliente', 'RF', 'FIC', 'RV_ALT',
        'RF_cumple', 'FIC_cumple', 'RV_ALT_cumple', 'cumple_perfil'
    ]
    
    if 'Alternativo' in estado_clientes_actual.columns:
        cols_orden.insert(5, 'Alternativo')
    if 'RV' in estado_clientes_actual.columns:
        cols_orden.insert(5, 'RV')
    
    estado_clientes_actual = estado_clientes_actual[cols_orden]
    
    # Lista activos preferidos por macroactivo
    activos_caracteristicas = df_portafolio[[
        'macroactivo', 'activo', 'perfil_activo', 'puntaje_activo',
        'rentabilidad_esperada_activo', 'volatilidad_activo'
    ]].drop_duplicates()
    
    # Ordenar por macroactivo y puntaje (mejor primero)
    activos_preferidos = activos_caracteristicas.sort_values(
        by=['macroactivo', 'puntaje_activo'],
        ascending=[True, False]
    ).reset_index(drop=True)
    
    # Calcular Sharpe individual por activo (rf=0)
    activos_preferidos['sharpe_activo'] = (
        activos_preferidos['rentabilidad_esperada_activo'] /
        activos_preferidos['volatilidad_activo']
    )
    
    return estado_clientes_actual, activos_preferidos

def analizar_y_rebalancear_portafolios(df_portafolio, df_estado_actual, df_activos_preferidos, limites):
    """
    Analiza portafolios y genera recomendaciones de rebalanceo.
    
    Estrategias:
    - CORRECCIÓN: Para clientes alertados por límites de perfil
    - OPTIMIZACIÓN: Para clientes que cumplen límites pero tienen activos de baja calidad
    
    Args:
        df_portafolio: DataFrame con posiciones de todos los clientes
        df_estado_actual: DataFrame con participaciones agregadas por macroactivo
        df_activos_preferidos: DataFrame con activos rankeados por calidad
        limites: dict con límites por perfil de riesgo
    
    Returns:
        DataFrame con recomendaciones de rebalanceo
    """
    todos_clientes = df_estado_actual['cliente_id'].tolist()
    recomendaciones_lista = []
    
    for cliente_id in todos_clientes:
        estado_cliente = df_estado_actual[df_estado_actual['cliente_id'] == cliente_id].iloc[0]
        perfil = estado_cliente['perfil_riesgo_cliente']
        limites_perfil = limites[perfil]
        posiciones_cliente = df_portafolio[df_portafolio['cliente_id'] == cliente_id][['activo', 'macroactivo', 'participacion_actual']].copy()
        
        # Identificar alertas de límites
        alertas = {}
        if estado_cliente['RF'] < limites_perfil['RF']['min']:
            alertas['RF'] = {'tipo': 'bajo', 'actual': estado_cliente['RF'], 'limite': limites_perfil['RF']['min'], 'diferencia': limites_perfil['RF']['min'] - estado_cliente['RF']}
        elif estado_cliente['RF'] > limites_perfil['RF']['max']:
            alertas['RF'] = {'tipo': 'alto', 'actual': estado_cliente['RF'], 'limite': limites_perfil['RF']['max'], 'diferencia': estado_cliente['RF'] - limites_perfil['RF']['max']}
        if estado_cliente['FIC'] < limites_perfil['FIC']['min']:
            alertas['FIC'] = {'tipo': 'bajo', 'actual': estado_cliente['FIC'], 'limite': limites_perfil['FIC']['min'], 'diferencia': limites_perfil['FIC']['min'] - estado_cliente['FIC']}
        elif estado_cliente['FIC'] > limites_perfil['FIC']['max']:
            alertas['FIC'] = {'tipo': 'alto', 'actual': estado_cliente['FIC'], 'limite': limites_perfil['FIC']['max'], 'diferencia': estado_cliente['FIC'] - limites_perfil['FIC']['max']}
        if estado_cliente['RV_ALT'] < limites_perfil['RV_ALT']['min']:
            alertas['RV_ALT'] = {'tipo': 'bajo', 'actual': estado_cliente['RV_ALT'], 'limite': limites_perfil['RV_ALT']['min'], 'diferencia': limites_perfil['RV_ALT']['min'] - estado_cliente['RV_ALT']}
        elif estado_cliente['RV_ALT'] > limites_perfil['RV_ALT']['max']:
            alertas['RV_ALT'] = {'tipo': 'alto', 'actual': estado_cliente['RV_ALT'], 'limite': limites_perfil['RV_ALT']['max'], 'diferencia': estado_cliente['RV_ALT'] - limites_perfil['RV_ALT']['max']}
        
        if len(alertas) > 0:
            # ESTRATEGIA CORRECCIÓN: Ajustar alertas de límites
            estrategia = 'CORRECCIÓN'
            
            # Reducir macroactivos que exceden límites
            for macro, info in alertas.items():
                if info['tipo'] == 'alto':
                    activos_reducir = posiciones_cliente[posiciones_cliente['macroactivo'] == macro].copy()
                    activos_reducir = activos_reducir.merge(df_activos_preferidos[['activo', 'puntaje_activo', 'sharpe_activo']], on='activo', how='left')
                    activos_reducir = activos_reducir.sort_values('puntaje_activo', ascending=True)
                    
                    reduccion_requerida = info['diferencia']
                    reduccion_acumulada = 0
                    for _, activo in activos_reducir.iterrows():
                        if reduccion_acumulada >= reduccion_requerida:
                            break
                        reduccion_posible = min(activo['participacion_actual'], reduccion_requerida - reduccion_acumulada)
                        recomendaciones_lista.append({
                            'cliente_id': cliente_id, 'perfil_riesgo': perfil, 'estrategia': estrategia, 'accion': 'REDUCIR',
                            'macroactivo': macro, 'activo': activo['activo'], 'participacion_actual': activo['participacion_actual'],
                            'ajuste_sugerido': -reduccion_posible, 'participacion_nueva': activo['participacion_actual'] - reduccion_posible,
                            'puntaje_activo': activo['puntaje_activo'], 'sharpe_activo': activo['sharpe_activo'],
                            'justificacion': f"Reducir {macro} de {info['actual']*100:.1f}% a {info['limite']*100:.1f}%"
                        })
                        reduccion_acumulada += reduccion_posible
            
            # Aumentar macroactivos por debajo de límites
            for macro, info in alertas.items():
                if info['tipo'] == 'bajo':
                    mejores_activos = df_activos_preferidos[df_activos_preferidos['macroactivo'] == macro].head(3)
                    aumento_requerido = info['diferencia']
                    for _, mejor_activo in mejores_activos.iterrows():
                        activo_existente = posiciones_cliente[posiciones_cliente['activo'] == mejor_activo['activo']]
                        if not activo_existente.empty:
                            participacion_actual = activo_existente.iloc[0]['participacion_actual']
                            recomendaciones_lista.append({'cliente_id': cliente_id, 'perfil_riesgo': perfil, 'estrategia': estrategia, 'accion': 'AUMENTAR', 'macroactivo': macro, 'activo': mejor_activo['activo'], 'participacion_actual': participacion_actual, 'ajuste_sugerido': aumento_requerido, 'participacion_nueva': participacion_actual + aumento_requerido, 'puntaje_activo': mejor_activo['puntaje_activo'], 'sharpe_activo': mejor_activo['sharpe_activo'], 'justificacion': f"Aumentar {macro} de {info['actual']*100:.1f}% a {info['limite']*100:.1f}%"})
                            break
                        else:
                            recomendaciones_lista.append({'cliente_id': cliente_id, 'perfil_riesgo': perfil, 'estrategia': estrategia, 'accion': 'AGREGAR', 'macroactivo': macro, 'activo': mejor_activo['activo'], 'participacion_actual': 0.0, 'ajuste_sugerido': aumento_requerido, 'participacion_nueva': aumento_requerido, 'puntaje_activo': mejor_activo['puntaje_activo'], 'sharpe_activo': mejor_activo['sharpe_activo'], 'justificacion': f"Agregar {macro} para alcanzar {info['limite']*100:.1f}%"})
                            break
        else:
            # ESTRATEGIA OPTIMIZACIÓN: Mejorar calidad de activos
            estrategia = 'OPTIMIZACIÓN'
            posiciones_calidad = posiciones_cliente.merge(df_activos_preferidos[['activo', 'puntaje_activo', 'sharpe_activo']], on='activo', how='left')
            activos_bajos = posiciones_calidad[posiciones_calidad['puntaje_activo'] < 3.5].copy()
            
            if len(activos_bajos) > 0:
                for _, activo_bajo in activos_bajos.iterrows():
                    macro = activo_bajo['macroactivo']
                    mejores_alternativas = df_activos_preferidos[(df_activos_preferidos['macroactivo'] == macro) & (~df_activos_preferidos['activo'].isin(posiciones_cliente['activo']))].head(1)
                    if not mejores_alternativas.empty:
                        mejor = mejores_alternativas.iloc[0]
                        recomendaciones_lista.append({'cliente_id': cliente_id, 'perfil_riesgo': perfil, 'estrategia': estrategia, 'accion': 'REDUCIR', 'macroactivo': macro, 'activo': activo_bajo['activo'], 'participacion_actual': activo_bajo['participacion_actual'], 'ajuste_sugerido': -activo_bajo['participacion_actual'], 'participacion_nueva': 0.0, 'puntaje_activo': activo_bajo['puntaje_activo'], 'sharpe_activo': activo_bajo['sharpe_activo'], 'justificacion': f"Reemplazar (puntaje={activo_bajo['puntaje_activo']:.2f})"})
                        recomendaciones_lista.append({'cliente_id': cliente_id, 'perfil_riesgo': perfil, 'estrategia': estrategia, 'accion': 'AGREGAR', 'macroactivo': macro, 'activo': mejor['activo'], 'participacion_actual': 0.0, 'ajuste_sugerido': activo_bajo['participacion_actual'], 'participacion_nueva': activo_bajo['participacion_actual'], 'puntaje_activo': mejor['puntaje_activo'], 'sharpe_activo': mejor['sharpe_activo'], 'justificacion': f"Mejora calidad (punto={mejor['puntaje_activo']:.2f})"})
    
    return pd.DataFrame(recomendaciones_lista)

def calcular_portafolio_propuesto(df_portafolio, df_recomendaciones):
    """
    Aplica las recomendaciones al portafolio actual y calcula las nuevas métricas.
    
    Returns:
        DataFrame con métricas propuestas por cliente
    """
    metricas_propuestas = []
    
    # Obtener clientes únicos y sus métricas actuales
    clientes_info = df_portafolio[['cliente_id', 'perfil_riesgo_cliente', 
                                     'rentabilidad_portafolio', 'volatilidad_portafolio']].drop_duplicates()
    
    for _, cliente_info in clientes_info.iterrows():
        cliente_id = cliente_info['cliente_id']
        
        # Portafolio actual del cliente
        port_actual = df_portafolio[df_portafolio['cliente_id'] == cliente_id].copy()
        
        # Verificar si hay recomendaciones para este cliente
        recs_cliente = df_recomendaciones[df_recomendaciones['cliente_id'] == cliente_id] if len(df_recomendaciones) > 0 else pd.DataFrame()
        
        if len(recs_cliente) == 0:
            # Sin recomendaciones: métricas propuestas = métricas actuales
            metricas_propuestas.append({
                'cliente_id': cliente_id,
                'perfil_riesgo': cliente_info['perfil_riesgo_cliente'],
                'rentabilidad_propuesta': cliente_info['rentabilidad_portafolio'],
                'volatilidad_propuesta': cliente_info['volatilidad_portafolio'],
                'sharpe_propuesto': cliente_info['rentabilidad_portafolio'] / cliente_info['volatilidad_portafolio'],
                'cambios_aplicados': False
            })
        else:
            # Aplicar recomendaciones
            # Crear un diccionario de participaciones propuestas
            participaciones_propuestas = {}
            
            # Inicializar con participaciones actuales
            for _, activo in port_actual.iterrows():
                participaciones_propuestas[activo['activo']] = activo['participacion_actual']
            
            # Aplicar ajustes de las recomendaciones
            for _, rec in recs_cliente.iterrows():
                if rec['accion'] == 'REDUCIR':
                    if rec['activo'] in participaciones_propuestas:
                        participaciones_propuestas[rec['activo']] = rec['participacion_nueva']
                elif rec['accion'] == 'AUMENTAR':
                    if rec['activo'] in participaciones_propuestas:
                        participaciones_propuestas[rec['activo']] = rec['participacion_nueva']
                elif rec['accion'] == 'AGREGAR':
                    participaciones_propuestas[rec['activo']] = rec['participacion_nueva']
            
            # Eliminar activos con participación 0
            participaciones_propuestas = {k: v for k, v in participaciones_propuestas.items() if v > 0.0001}
            
            # Normalizar para que sume 100%
            total_part = sum(participaciones_propuestas.values())
            participaciones_propuestas = {k: v/total_part for k, v in participaciones_propuestas.items()}
            
            # Calcular nuevas métricas ponderadas
            rentabilidad_prop = 0
            volatilidad_prop = 0
            
            for activo, participacion in participaciones_propuestas.items():
                activo_data = df_portafolio[df_portafolio['activo'] == activo].iloc[0]
                rentabilidad_prop += participacion * activo_data['rentabilidad_esperada_activo']
                volatilidad_prop += (participacion ** 2) * (activo_data['volatilidad_activo'] ** 2)
            
            volatilidad_prop = np.sqrt(volatilidad_prop)
            sharpe_prop = rentabilidad_prop / volatilidad_prop if volatilidad_prop > 0 else 0
            
            metricas_propuestas.append({
                'cliente_id': cliente_id,
                'perfil_riesgo': cliente_info['perfil_riesgo_cliente'],
                'rentabilidad_propuesta': rentabilidad_prop,
                'volatilidad_propuesta': volatilidad_prop,
                'sharpe_propuesto': sharpe_prop,
                'cambios_aplicados': True
            })
    
    return pd.DataFrame(metricas_propuestas)

def graficar_comparativa_clientes(df_portafolio, df_estado_actual, metricas_propuestas, df_recomendaciones):
    """
    Crea una grilla de 10 gráficos (uno por cliente) mostrando:
    - Actual vs Propuesto en Riesgo-Retorno
    - Mejora en Sharpe Ratio
    - Cumplimiento de límites
    """
    
    # Obtener métricas actuales
    metricas_actual = df_portafolio[['cliente_id', 'rentabilidad_portafolio', 
                                       'volatilidad_portafolio', 'perfil_riesgo_cliente']].drop_duplicates()
    metricas_actual['sharpe_actual'] = metricas_actual['rentabilidad_portafolio'] / metricas_actual['volatilidad_portafolio']
    
    # Merge con estado para cumplimiento
    metricas_actual = metricas_actual.merge(df_estado_actual[['cliente_id', 'cumple_perfil']], on='cliente_id')
    
    # Merge con métricas propuestas
    comparativa = metricas_actual.merge(
        metricas_propuestas[['cliente_id', 'rentabilidad_propuesta', 'volatilidad_propuesta', 'sharpe_propuesto', 'cambios_aplicados']],
        on='cliente_id'
    )
    
    # Ordenar por cliente_id
    comparativa = comparativa.sort_values('cliente_id')
    
    # Colores por perfil
    color_map = {
        'Conservador': '#3b82f6',
        'Moderado': '#f59e0b',
        'Arriesgado': '#ef4444'
    }
    
    # Crear figura con subplots (2 filas × 5 columnas)
    fig, axes = plt.subplots(2, 5, figsize=(22, 10), facecolor='white')
    fig.suptitle('Comparativa ACTUAL vs PROPUESTO — Portafolios por Cliente', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    for idx, (_, cliente) in enumerate(comparativa.iterrows()):
        ax = axes[idx]
        ax.set_facecolor('#f8fafc')
        
        cliente_id = int(cliente['cliente_id'])
        perfil = cliente['perfil_riesgo_cliente']
        color = color_map[perfil]
        
        # PUNTO ACTUAL
        ax.scatter(
            cliente['volatilidad_portafolio'] * 100,
            cliente['rentabilidad_portafolio'] * 100,
            s=300,
            c=color,
            marker='o',
            edgecolors='white',
            linewidths=2,
            alpha=0.6,
            label='Actual',
            zorder=3
        )
        
        # PUNTO PROPUESTO
        ax.scatter(
            cliente['volatilidad_propuesta'] * 100,
            cliente['rentabilidad_propuesta'] * 100,
            s=300,
            c=color,
            marker='*',
            edgecolors='black',
            linewidths=1.5,
            alpha=0.9,
            label='Propuesto',
            zorder=4
        )
        
        # FLECHA DE MOVIMIENTO (si hay cambios)
        if cliente['cambios_aplicados']:
            ax.annotate(
                '',
                xy=(cliente['volatilidad_propuesta'] * 100, cliente['rentabilidad_propuesta'] * 100),
                xytext=(cliente['volatilidad_portafolio'] * 100, cliente['rentabilidad_portafolio'] * 100),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=color, alpha=0.6)
            )
        
        # Línea de Sharpe actual y propuesto
        vol_range = np.linspace(0, max(cliente['volatilidad_portafolio'], cliente['volatilidad_propuesta']) * 120, 50)
        ax.plot(vol_range, cliente['sharpe_actual'] * vol_range, 'k--', alpha=0.3, linewidth=1, label=f"Sharpe Act={cliente['sharpe_actual']:.2f}")
        ax.plot(vol_range, cliente['sharpe_propuesto'] * vol_range, 'g--', alpha=0.4, linewidth=1, label=f"Sharpe Prop={cliente['sharpe_propuesto']:.2f}")
        
        # Título del subplot
        cumple_emoji = "✓" if cliente['cumple_perfil'] else "✗"
        mejora_sharpe = cliente['sharpe_propuesto'] - cliente['sharpe_actual']
        mejora_color = '#16a34a' if mejora_sharpe > 0 else '#dc2626'
        
        ax.set_title(
            f"Cliente {cliente_id} ({perfil[:3]}.) {cumple_emoji}\nΔSharpe: {mejora_sharpe:+.3f}",
            fontsize=10,
            fontweight='bold',
            color=mejora_color if abs(mejora_sharpe) > 0.01 else '#64748b'
        )
        
        # Configuración de ejes
        ax.set_xlabel('Vol (%)', fontsize=8)
        ax.set_ylabel('Ret (%)', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        # Leyenda pequeña
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left', framealpha=0.8)
    
    
    plt.tight_layout()
    
    # Estadísticas de mejora
    comparativa['mejora_sharpe'] = comparativa['sharpe_propuesto'] - comparativa['sharpe_actual']
    comparativa['mejora_rentabilidad'] = comparativa['rentabilidad_propuesta'] - comparativa['rentabilidad_portafolio']
    comparativa['reduccion_volatilidad'] = comparativa['volatilidad_portafolio'] - comparativa['volatilidad_propuesta']
    
    return comparativa, fig

# ============================================================================
# SECCIÓN DASHBOARD DE STREAMLIT
# ============================================================================

# Función para cargar y procesar datos (con caché)
@st.cache_data
def cargar_y_procesar_datos():
    """Carga el CSV y ejecuta todo el pipeline de análisis"""
    # Cargar datos (ruta robusta que funciona en cualquier entorno)
    csv_path = os.path.join(os.path.dirname(__file__),'portafolios.csv')
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Paso 1: Preparar datos
    estado, activos = preparar_datos_portafolio(df, limites_perfiles)
    
    # Paso 2: Analizar y generar recomendaciones
    recs = analizar_y_rebalancear_portafolios(df, estado, activos, limites_perfiles)
    
    # Paso 3: Calcular métricas propuestas
    metricas = calcular_portafolio_propuesto(df, recs)
    
    # Paso 4: Generar comparativa
    comparativa, fig = graficar_comparativa_clientes(df, estado, metricas, recs)
    
    return df, estado, activos, recs, metricas, comparativa, fig

# Cargar datos
with st.spinner('⏳ Cargando y procesando datos...'):
    df_portafolio, estado_clientes, activos_preferidos, recomendaciones, metricas_propuestas, comparativa, fig_comparativa = cargar_y_procesar_datos()

# ============================================================================
# HEADER & TÍTULO
# ============================================================================
st.title('Análisis & Rebalanceo de Portafolios')

st.divider()

# ============================================================================
# PIPELINE VISUAL
# ============================================================================
st.subheader('Pipeline de Análisis')

col_pipe1, col_pipe2, col_pipe3, col_pipe4 = st.columns(4)

with col_pipe1:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white;'>
        <h3 style='margin: 0; font-size: 2em;'>①</h3>
        <p style='margin: 5px 0; font-weight: bold;'>Preparar Datos</p>
        <p style='margin: 0; font-size: 0.85em;'>Agregación y validación</p>
    </div>
    """, unsafe_allow_html=True)

with col_pipe2:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                border-radius: 10px; color: white;'>
        <h3 style='margin: 0; font-size: 2em;'>②</h3>
        <p style='margin: 5px 0; font-weight: bold;'>Analizar & Rebalancear</p>
        <p style='margin: 0; font-size: 0.85em;'>Generar recomendaciones</p>
    </div>
    """, unsafe_allow_html=True)

with col_pipe3:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                border-radius: 10px; color: white;'>
        <h3 style='margin: 0; font-size: 2em;'>③</h3>
        <p style='margin: 5px 0; font-weight: bold;'>Calcular Propuesto</p>
        <p style='margin: 0; font-size: 0.85em;'>Nuevas métricas</p>
    </div>
    """, unsafe_allow_html=True)

with col_pipe4:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                border-radius: 10px; color: white;'>
        <h3 style='margin: 0; font-size: 2em;'>④</h3>
        <p style='margin: 5px 0; font-weight: bold;'>Visualizar</p>
        <p style='margin: 0; font-size: 0.85em;'>Comparativa final</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# KPIs PRINCIPALES
# ============================================================================
st.subheader('Métricas Clave del Análisis')

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label='Total Clientes',
        value=len(estado_clientes),
        delta=None
    )

with col2:
    clientes_cumplen = estado_clientes['cumple_perfil'].sum()
    porcentaje_cumple = (clientes_cumplen / len(estado_clientes)) * 100
    st.metric(
        label='Cumplen Límites',
        value=f'{clientes_cumplen}',
        delta=f'{porcentaje_cumple:.0f}%'
    )

with col3:
    mejora_sharpe_avg = comparativa['mejora_sharpe'].mean()
    st.metric(
        label='Mejora Sharpe Promedio',
        value=f'{mejora_sharpe_avg:+.4f}',
        delta='Mejora' if mejora_sharpe_avg > 0 else 'Sin cambio'
    )

with col4:
    mejora_rent_avg = comparativa['mejora_rentabilidad'].mean() * 100
    st.metric(
        label='Mejora Rentabilidad',
        value=f'{mejora_rent_avg:+.2f}%',
        delta='Mejora' if mejora_rent_avg > 0 else 'Sin cambio'
    )

st.divider()

# ============================================================================
# DISTRIBUCIONES: PIE CHARTS
# ============================================================================
st.subheader('Distribución de Portafolios')

col_pie1, col_pie2, col_pie3 = st.columns(3)

# PIE 1: Distribución por perfil
with col_pie1:
    st.markdown('**Clientes por Perfil de Riesgo**')
    perfil_counts = estado_clientes['perfil_riesgo_cliente'].value_counts()
    
    fig_pie1, ax_pie1 = plt.subplots(figsize=(6, 6))
    colors_perfil = ['#3b82f6', '#f59e0b', '#ef4444']
    wedges, texts, autotexts = ax_pie1.pie(
        perfil_counts.values,
        labels=perfil_counts.index,
        autopct='%1.0f%%',
        startangle=90,
        colors=colors_perfil,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    for autotext in autotexts:
        autotext.set_color('white')
    ax_pie1.set_title('Distribución por Perfil', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig_pie1)

# PIE 2: Cumplimiento de límites
with col_pie2:
    st.markdown('**Cumplimiento de Límites**')
    cumple_counts = estado_clientes['cumple_perfil'].value_counts()
    
    fig_pie2, ax_pie2 = plt.subplots(figsize=(6, 6))
    colors_cumple = ['#16a34a', '#dc2626']
    labels_cumple = ['Cumple' if x else 'Alertado' for x in cumple_counts.index]
    wedges, texts, autotexts = ax_pie2.pie(
        cumple_counts.values,
        labels=labels_cumple,
        autopct='%1.0f%%',
        startangle=90,
        colors=colors_cumple,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    for autotext in autotexts:
        autotext.set_color('white')
    ax_pie2.set_title('Estado de Cumplimiento', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig_pie2)

# PIE 3: Estrategias aplicadas
with col_pie3:
    st.markdown('**Estrategias de Rebalanceo**')
    if len(recomendaciones) > 0:
        estrategia_counts = recomendaciones['estrategia'].value_counts()
        
        fig_pie3, ax_pie3 = plt.subplots(figsize=(6, 6))
        colors_estrategia = ['#8b5cf6', '#06b6d4']
        wedges, texts, autotexts = ax_pie3.pie(
            estrategia_counts.values,
            labels=estrategia_counts.index,
            autopct='%1.0f%%',
            startangle=90,
            colors=colors_estrategia,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_color('white')
        ax_pie3.set_title('Distribución de Estrategias', fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig_pie3)
    else:
        st.info('Todos los portafolios están óptimos')

st.divider()

# ============================================================================
# METODOLOGÍA: CÁLCULOS Y NORMALIZACIÓN
# ============================================================================
st.subheader('Metodología de Optimización')

# Card principal con explicación
st.markdown("""
<div style='padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h3 style='margin: 0 0 15px 0; font-size: 1.4em;'>🧮 Proceso de Cálculo y Rebalanceo</h3>
    <p style='margin: 0 0 10px 0; font-size: 1.05em; line-height: 1.6;'>
        El modelo optimiza portafolios mediante métricas cuantitativas robustas y un proceso dinámico de normalización.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# Métricas en 3 columnas
col_met1, col_met2, col_met3 = st.columns(3)

with col_met1:
    st.markdown("""
    <div style='padding: 20px; background: white; border-left: 5px solid #3b82f6; 
                border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h4 style='margin: 0 0 10px 0; color: #1e40af;'>📊 Rentabilidad Portafolio</h4>
        <p style='margin: 0; color: #475569; font-size: 0.95em; line-height: 1.5;'>
            <strong>Fórmula:</strong> Suma ponderada de retornos individuales
        </p>
        <div style='background: #f1f5f9; padding: 10px; border-radius: 5px; margin-top: 10px; font-family: monospace;'>
            R<sub>p</sub> = Σ (w<sub>i</sub> × r<sub>i</sub>)
        </div>
        <p style='margin: 10px 0 0 0; color: #64748b; font-size: 0.85em;'>
            Donde w<sub>i</sub> = participación del activo i<br>
            r<sub>i</sub> = rentabilidad esperada del activo i
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_met2:
    st.markdown("""
    <div style='padding: 20px; background: white; border-left: 5px solid #f59e0b; 
                border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h4 style='margin: 0 0 10px 0; color: #d97706;'>📉 Volatilidad Portafolio</h4>
        <p style='margin: 0; color: #475569; font-size: 0.95em; line-height: 1.5;'>
            <strong>Fórmula:</strong> Raíz de suma ponderada de varianzas
        </p>
        <div style='background: #fef3c7; padding: 10px; border-radius: 5px; margin-top: 10px; font-family: monospace;'>
            σ<sub>p</sub> = √[Σ (w<sub>i</sub>² × σ<sub>i</sub>²)]
        </div>
        <p style='margin: 10px 0 0 0; color: #64748b; font-size: 0.85em;'>
            Donde w<sub>i</sub> = participación del activo i<br>
            σ<sub>i</sub> = volatilidad del activo i
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_met3:
    st.markdown("""
    <div style='padding: 20px; background: white; border-left: 5px solid #10b981; 
                border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h4 style='margin: 0 0 10px 0; color: #059669;'>⚡ Sharpe Ratio</h4>
        <p style='margin: 0; color: #475569; font-size: 0.95em; line-height: 1.5;'>
            <strong>Fórmula:</strong> Eficiencia riesgo-retorno
        </p>
        <div style='background: #d1fae5; padding: 10px; border-radius: 5px; margin-top: 10px; font-family: monospace;'>
            Sharpe = (R<sub>p</sub> - R<sub>f</sub>) / σ<sub>p</sub>
        </div>
        <p style='margin: 10px 0 0 0; color: #64748b; font-size: 0.85em;'>
            Donde R<sub>f</sub> = tasa libre de riesgo
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# Flujo de rebalanceo
st.markdown("""
<div style='padding: 20px; background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 10px; margin-top: 15px;'>
    <h4 style='margin: 0 0 15px 0; color: #1e293b;'>🔄 Proceso Dinámico de Rebalanceo y Normalización</h4>
</div>
""", unsafe_allow_html=True)

# Pasos del proceso usando st.markdown para cada uno
col_flow1, col_flow2 = st.columns([1, 20])

with col_flow1:
    st.markdown("<div style='background: #6366f1; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1em;'>1</div>", unsafe_allow_html=True)
with col_flow2:
    st.markdown("**Detectar Alertas:** Identificar macroactivos fuera de límites (RF, FIC, RV_ALT) según perfil de riesgo")

col_flow3, col_flow4 = st.columns([1, 20])
with col_flow3:
    st.markdown("<div style='background: #8b5cf6; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1em;'>2</div>", unsafe_allow_html=True)
with col_flow4:
    st.markdown("**Aplicar Ajustes:** Reducir activos de baja calidad (bajo puntaje) si excede límites / Aumentar mejores activos si está por debajo")

col_flow5, col_flow6 = st.columns([1, 20])
with col_flow5:
    st.markdown("<div style='background: #06b6d4; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1em;'>3</div>", unsafe_allow_html=True)
with col_flow6:
    st.markdown("**Normalización Dinámica:** Recalcular participaciones para que sumen 100%: `w'ᵢ = wᵢ / Σwⱼ`")

col_flow7, col_flow8 = st.columns([1, 20])
with col_flow7:
    st.markdown("<div style='background: #10b981; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1em;'>4</div>", unsafe_allow_html=True)
with col_flow8:
    st.markdown("**Recalcular Métricas:** Con las nuevas ponderaciones w'ᵢ, calcular Rₚ, σₚ, y Sharpe propuesto")

col_flow9, col_flow10 = st.columns([1, 20])
with col_flow9:
    st.markdown("<div style='background: #f59e0b; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1em;'>5</div>", unsafe_allow_html=True)
with col_flow10:
    st.markdown("**Validar Mejora:** Comparar métricas propuestas vs actuales: Δ Sharpe, Δ Rentabilidad, Δ Volatilidad")

# Ejemplo de normalización
st.info("""
**💡 Ejemplo de Normalización:**

Si un portafolio tiene [A: 40%, B: 35%, C: 25%] y reducimos C en 10%, 
las nuevas participaciones sin normalizar serían [40%, 35%, 15%] = **90% total**.

**Tras normalizar:** [40/90 = 44.4%, 35/90 = 38.9%, 15/90 = 16.7%] = **100% ✓**
""")

st.markdown("")

# Tarjeta de estrategias
col_est1, col_est2 = st.columns(2)

with col_est1:
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                border-radius: 10px; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 160px;'>
        <h4 style='margin: 0 0 10px 0; font-size: 1.2em;'>⚠️ CORRECCIÓN</h4>
        <p style='margin: 0; font-size: 0.95em; line-height: 1.6;'>
            <strong>Cuándo:</strong> Cliente incumple límites de su perfil de riesgo<br>
            <strong>Acción:</strong> Ajustar macroactivos (RF, FIC, RV_ALT) para cumplir rangos regulatorios
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_est2:
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); 
                border-radius: 10px; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 160px;'>
        <h4 style='margin: 0 0 10px 0; font-size: 1.2em;'>⚡ OPTIMIZACIÓN</h4>
        <p style='margin: 0; font-size: 0.95em; line-height: 1.6;'>
            <strong>Cuándo:</strong> Cliente cumple límites pero tiene activos de baja calidad<br>
            <strong>Acción:</strong> Reemplazar activos con puntaje <3.5 por mejores alternativas
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# IMPACTO DEL MODELO: MEJORAS CUANTIFICADAS
# ============================================================================
st.subheader('Impacto del Modelo de Optimización')

# Explicación de métricas
st.info("""
**Guía de Interpretación:**
- **Sharpe Ratio**: Mayor es mejor (más retorno por unidad de riesgo).
- **Rentabilidad**: Positivo = aumentó | Negativo = se sacrificó para reducir riesgo.
- **Volatilidad**: Reducción positiva = menos riesgo | Negativa = más riesgo.

**Trade-off común:** El modelo puede sacrificar rentabilidad para cumplir límites regulatorios y reducir volatilidad, mejorando la eficiencia (Sharpe).
""")

# Preparar datos ordenados
comparativa_ordenada = comparativa.sort_values('cliente_id').copy()
comparativa_ordenada['cliente_label'] = comparativa_ordenada['cliente_id'].astype(int).astype(str)

col_imp1, col_imp2 = st.columns(2)

# Mejora en Sharpe Ratio
with col_imp1:
    st.markdown('**Mejora en Sharpe Ratio**')
    fig_sharpe, ax_sharpe = plt.subplots(figsize=(8, 5))
    
    colores_sharpe = ['#16a34a' if x > 0 else '#dc2626' for x in comparativa_ordenada['mejora_sharpe']]
    
    bars = ax_sharpe.barh(
        comparativa_ordenada['cliente_label'],
        comparativa_ordenada['mejora_sharpe'],
        color=colores_sharpe,
        alpha=0.8
    )
    
    ax_sharpe.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax_sharpe.set_xlabel('Mejora en Sharpe Ratio', fontweight='bold')
    ax_sharpe.set_ylabel('Cliente', fontweight='bold')
    ax_sharpe.grid(axis='x', alpha=0.3)
    ax_sharpe.invert_yaxis()
    
    # Anotar valores
    for i, (idx, row) in enumerate(comparativa_ordenada.iterrows()):
        valor = row['mejora_sharpe']
        ax_sharpe.text(
            valor + 0.002 if valor > 0 else valor - 0.002,
            i,
            f'{valor:+.3f}',
            va='center',
            ha='left' if valor > 0 else 'right',
            fontsize=8,
            fontweight='bold'
        )
    
    plt.tight_layout()
    st.pyplot(fig_sharpe)
    
    mejoras_positivas = (comparativa_ordenada['mejora_sharpe'] > 0).sum()
    st.success(f'**{mejoras_positivas}/{len(comparativa_ordenada)} clientes** mejoraron su Sharpe (más eficientes)')

# Mejora en Rentabilidad
with col_imp2:
    st.markdown('**Mejora en Rentabilidad Esperada**')
    fig_rent, ax_rent = plt.subplots(figsize=(8, 5))
    
    mejora_rent_pct = comparativa_ordenada['mejora_rentabilidad'] * 100
    colores_rent = ['#16a34a' if x > 0 else '#dc2626' for x in mejora_rent_pct]
    
    bars = ax_rent.barh(
        comparativa_ordenada['cliente_label'],
        mejora_rent_pct,
        color=colores_rent,
        alpha=0.8
    )
    
    ax_rent.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax_rent.set_xlabel('Mejora en Rentabilidad (%)', fontweight='bold')
    ax_rent.set_ylabel('Cliente', fontweight='bold')
    ax_rent.grid(axis='x', alpha=0.3)
    ax_rent.invert_yaxis()
    
    # Anotar valores
    for i, (idx, row) in enumerate(comparativa_ordenada.iterrows()):
        valor = row['mejora_rentabilidad'] * 100
        ax_rent.text(
            valor + 0.05 if valor > 0 else valor - 0.05,
            i,
            f'{valor:+.2f}%',
            va='center',
            ha='left' if valor > 0 else 'right',
            fontsize=8,
            fontweight='bold'
        )
    
    plt.tight_layout()
    st.pyplot(fig_rent)
    
    rent_sacrificios = (comparativa_ordenada['mejora_rentabilidad'] < 0).sum()
    if rent_sacrificios > 0:
        st.warning(f'**{rent_sacrificios} clientes** sacrificaron rentabilidad para cumplir límites y reducir riesgo')

st.divider()

# Reducción de Volatilidad + Scatter
col_imp3, col_imp4 = st.columns(2)

with col_imp3:
    st.markdown('**Reducción de Volatilidad (Riesgo)**')
    fig_vol, ax_vol = plt.subplots(figsize=(8, 5))
    
    reduccion_vol_pct = comparativa_ordenada['reduccion_volatilidad'] * 100
    colores_vol = ['#16a34a' if x > 0 else '#dc2626' for x in reduccion_vol_pct]
    
    bars = ax_vol.barh(
        comparativa_ordenada['cliente_label'],
        reduccion_vol_pct,
        color=colores_vol,
        alpha=0.8
    )
    
    ax_vol.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax_vol.set_xlabel('Reducción de Volatilidad (%)', fontweight='bold')
    ax_vol.set_ylabel('Cliente', fontweight='bold')
    ax_vol.grid(axis='x', alpha=0.3)
    ax_vol.invert_yaxis()
    
    # Anotar valores
    for i, (idx, row) in enumerate(comparativa_ordenada.iterrows()):
        valor = row['reduccion_volatilidad'] * 100
        ax_vol.text(
            valor + 0.05 if valor > 0 else valor - 0.05,
            i,
            f'{valor:+.2f}%',
            va='center',
            ha='left' if valor > 0 else 'right',
            fontsize=8,
            fontweight='bold'
        )
    
    plt.tight_layout()
    st.pyplot(fig_vol)
    
    vol_reducciones = (comparativa_ordenada['reduccion_volatilidad'] > 0).sum()
    vol_aumentos = (comparativa_ordenada['reduccion_volatilidad'] < 0).sum()
    if vol_reducciones > 0:
        st.success(f'**{vol_reducciones}/{len(comparativa_ordenada)} clientes** redujeron riesgo (volatilidad)')
    if vol_aumentos > 0:
        st.error(f'**{vol_aumentos} clientes** aumentaron volatilidad')

with col_imp4:
    st.markdown('**Espacio Riesgo-Retorno: Actual → Propuesto**')
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    
    # Mapeo de colores por perfil
    color_map = {
        'Conservador': '#3b82f6',
        'Moderado': '#f59e0b',
        'Arriesgado': '#ef4444'
    }
    
    # Plotear ACTUAL (círculos)
    for perfil, color in color_map.items():
        mask = comparativa_ordenada['perfil_riesgo_cliente'] == perfil
        datos_perfil = comparativa_ordenada[mask]
        
        ax_scatter.scatter(
            datos_perfil['volatilidad_portafolio'] * 100,
            datos_perfil['rentabilidad_portafolio'] * 100,
            s=150,
            c=color,
            marker='o',
            alpha=0.5,
            label=f'{perfil} (Actual)',
            edgecolors='white',
            linewidths=1
        )
    
    # Plotear PROPUESTO (estrellas)
    for perfil, color in color_map.items():
        mask = comparativa_ordenada['perfil_riesgo_cliente'] == perfil
        datos_perfil = comparativa_ordenada[mask]
        
        ax_scatter.scatter(
            datos_perfil['volatilidad_propuesta'] * 100,
            datos_perfil['rentabilidad_propuesta'] * 100,
            s=200,
            c=color,
            marker='*',
            alpha=0.9,
            label=f'{perfil} (Propuesto)',
            edgecolors='black',
            linewidths=1
        )
    
    # Flechas de movimiento
    for idx, row in comparativa_ordenada.iterrows():
        if row['cambios_aplicados']:
            ax_scatter.annotate(
                '',
                xy=(row['volatilidad_propuesta'] * 100, row['rentabilidad_propuesta'] * 100),
                xytext=(row['volatilidad_portafolio'] * 100, row['rentabilidad_portafolio'] * 100),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6)
            )
    
    ax_scatter.set_xlabel('Volatilidad (%)', fontweight='bold')
    ax_scatter.set_ylabel('Rentabilidad Esperada (%)', fontweight='bold')
    ax_scatter.legend(fontsize=7, loc='best', ncol=2)
    ax_scatter.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_scatter)

st.divider()

# ============================================================================
# COMPARATIVA VISUAL COMPLETA (Grid 2x5)
# ============================================================================
st.subheader('Comparativa Detallada: Todos los Clientes')

st.pyplot(fig_comparativa)

st.divider()

# ============================================================================
# TABLA DE RESULTADOS
# ============================================================================
st.subheader('Resultados Detallados por Cliente')

# Preparar tabla
tabla_resultados = comparativa[['cliente_id', 'perfil_riesgo_cliente', 'cumple_perfil', 'cambios_aplicados', 
                                  'mejora_sharpe', 'mejora_rentabilidad', 'reduccion_volatilidad']].copy()
tabla_resultados['cliente_id'] = tabla_resultados['cliente_id'].astype(int)
tabla_resultados['cumple_perfil'] = tabla_resultados['cumple_perfil'].apply(lambda x: '✅ Sí' if x else '❌ No')
tabla_resultados['cambios_aplicados'] = tabla_resultados['cambios_aplicados'].apply(lambda x: '✅ Sí' if x else '➖ No')
tabla_resultados['mejora_sharpe'] = tabla_resultados['mejora_sharpe'].apply(lambda x: f'{x:+.4f}')
tabla_resultados['mejora_rentabilidad'] = tabla_resultados['mejora_rentabilidad'].apply(lambda x: f'{x*100:+.2f}%')
tabla_resultados['reduccion_volatilidad'] = tabla_resultados['reduccion_volatilidad'].apply(lambda x: f'{x*100:+.2f}%')

tabla_resultados.columns = ['Cliente', 'Perfil', 'Cumple Límites', 'Cambios Aplicados', 
                             'Δ Sharpe (Eficiencia)', 'Δ Rentabilidad', 'Δ Volatilidad (Reducción)']

st.dataframe(tabla_resultados, use_container_width=True, hide_index=True)

st.divider()

# ============================================================================
# RESUMEN EJECUTIVO FINAL
# ============================================================================
st.subheader('Resumen Ejecutivo')

col_resumen1, col_resumen2, col_resumen3 = st.columns(3)

with col_resumen1:
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white;'>
        <h4 style='margin: 0;'>Eficiencia (Sharpe Ratio)</h4>
        <p style='font-size: 1.5em; margin: 10px 0; font-weight: bold;'>{:+.4f}</p>
        <p style='margin: 0; font-size: 0.9em;'>Mejora promedio por cliente</p>
    </div>
    """.format(comparativa['mejora_sharpe'].mean()), unsafe_allow_html=True)

with col_resumen2:
    mejora_rent_prom = comparativa['mejora_rentabilidad'].mean()*100
    color_rent = '#16a34a' if mejora_rent_prom > 0 else '#dc2626'
    texto_rent = 'Aumentó en promedio' if mejora_rent_prom > 0 else 'Trade-off: Sacrificada para reducir riesgo'
    
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                border-radius: 10px; color: white;'>
        <h4 style='margin: 0;'>Rentabilidad Esperada</h4>
        <p style='font-size: 1.5em; margin: 10px 0; font-weight: bold; color: {};'>{:+.2f}%</p>
        <p style='margin: 0; font-size: 0.9em;'>{}</p>
    </div>
    """.format('white', mejora_rent_prom, texto_rent), unsafe_allow_html=True)

with col_resumen3:
    reduccion_vol_prom = comparativa['reduccion_volatilidad'].mean()*100
    
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                border-radius: 10px; color: white;'>
        <h4 style='margin: 0;'>Riesgo (Volatilidad)</h4>
        <p style='font-size: 1.5em; margin: 10px 0; font-weight: bold;'>{:+.2f}%</p>
        <p style='margin: 0; font-size: 0.9em;'>Reducción promedio</p>
    </div>
    """.format(reduccion_vol_prom), unsafe_allow_html=True)


# ============================================================================
# ANÁLISIS INDIVIDUAL (Opcional - Expandible)
# ============================================================================
st.divider()
with st.expander("Detalle por cliente"):
    
    # Selector de cliente
    clientes_lista = sorted(estado_clientes['cliente_id'].unique())
    cliente_seleccionado = st.selectbox(
        'Selecciona un cliente para análisis detallado:',
        clientes_lista,
        format_func=lambda x: f'Cliente {int(x)}'
    )
    
    # Obtener datos del cliente
    cliente_estado = estado_clientes[estado_clientes['cliente_id'] == cliente_seleccionado].iloc[0]
    cliente_metricas = comparativa[comparativa['cliente_id'] == cliente_seleccionado].iloc[0]
    cliente_recs = recomendaciones[recomendaciones['cliente_id'] == cliente_seleccionado]
    
    # Información básica
    st.subheader(f"Cliente {int(cliente_seleccionado)} — Perfil {cliente_estado['perfil_riesgo_cliente']}")
    
    # Estado de cumplimiento
    if cliente_estado['cumple_perfil']:
        st.success('Cumple con todos los límites de su perfil')
    else:
        st.error('Alertado por límites de su perfil — Requiere CORRECCIÓN')
    
    st.divider()
    
    # Métricas actual vs propuesto
    st.markdown('**Métricas: Actual vs Propuesto**')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label='Rentabilidad',
            value=f"{cliente_metricas['rentabilidad_propuesta']*100:.2f}%",
            delta=f"{cliente_metricas['mejora_rentabilidad']*100:+.2f}%",
            help='Actual → Propuesto'
        )
    
    with col2:
        st.metric(
            label='Volatilidad',
            value=f"{cliente_metricas['volatilidad_propuesta']*100:.2f}%",
            delta=f"{-cliente_metricas['reduccion_volatilidad']*100:+.2f}%",
            delta_color='inverse',
            help='Reducción es positivo'
        )
    
    with col3:
        st.metric(
            label='Sharpe Ratio',
            value=f"{cliente_metricas['sharpe_propuesto']:.3f}",
            delta=f"{cliente_metricas['mejora_sharpe']:+.3f}",
            help='Actual → Propuesto'
        )
    
    st.divider()
    
    # Asignación por macroactivo
    st.markdown('**Asignación por Macroactivo vs Límites**')
    
    # Crear DataFrame para visualización
    asignacion_data = {
        'Macroactivo': ['RF', 'FIC', 'RV_ALT'],
        'Actual (%)': [
            cliente_estado['RF'] * 100,
            cliente_estado['FIC'] * 100,
            cliente_estado['RV_ALT'] * 100
        ],
        'Límite Min (%)': [
            limites_perfiles[cliente_estado['perfil_riesgo_cliente']]['RF']['min'] * 100,
            limites_perfiles[cliente_estado['perfil_riesgo_cliente']]['FIC']['min'] * 100,
            limites_perfiles[cliente_estado['perfil_riesgo_cliente']]['RV_ALT']['min'] * 100
        ],
        'Límite Max (%)': [
            limites_perfiles[cliente_estado['perfil_riesgo_cliente']]['RF']['max'] * 100,
            limites_perfiles[cliente_estado['perfil_riesgo_cliente']]['FIC']['max'] * 100,
            limites_perfiles[cliente_estado['perfil_riesgo_cliente']]['RV_ALT']['max'] * 100
        ]
    }
    
    df_asignacion = pd.DataFrame(asignacion_data)
    
    # Gráfica de barras con límites
    fig_asig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(df_asignacion))
    width = 0.35
    
    # Barras de asignación actual
    bars = ax.bar(x, df_asignacion['Actual (%)'], width, label='Asignación Actual', 
                   color=['#3b82f6', '#f59e0b', '#ef4444'], alpha=0.7)
    
    # Líneas de límites
    for i, row in df_asignacion.iterrows():
        ax.hlines(row['Límite Min (%)'], i-width/2, i+width/2, colors='green', 
                  linestyles='--', linewidths=2, label='Límite Min' if i == 0 else '')
        ax.hlines(row['Límite Max (%)'], i-width/2, i+width/2, colors='red', 
                  linestyles='--', linewidths=2, label='Límite Max' if i == 0 else '')
    
    ax.set_xlabel('Macroactivo')
    ax.set_ylabel('Participación (%)')
    ax.set_title(f'Cliente {int(cliente_seleccionado)} - Asignación vs Límites')
    ax.set_xticks(x)
    ax.set_xticklabels(df_asignacion['Macroactivo'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_asig)
    
    st.divider()
    
    # Recomendaciones
    st.markdown('**Recomendaciones de Rebalanceo**')
    
    if len(cliente_recs) > 0:
        st.info(f"**{len(cliente_recs)} recomendaciones** — Estrategia: **{cliente_recs.iloc[0]['estrategia']}**")
        
        # Mostrar tabla de recomendaciones
        cols_mostrar = ['accion', 'macroactivo', 'activo', 'participacion_actual', 
                        'ajuste_sugerido', 'participacion_nueva', 'justificacion']
        df_recs_display = cliente_recs[cols_mostrar].copy()
        df_recs_display['participacion_actual'] = df_recs_display['participacion_actual'].apply(lambda x: f'{x*100:.2f}%')
        df_recs_display['ajuste_sugerido'] = df_recs_display['ajuste_sugerido'].apply(lambda x: f'{x*100:+.2f}%')
        df_recs_display['participacion_nueva'] = df_recs_display['participacion_nueva'].apply(lambda x: f'{x*100:.2f}%')
        
        st.dataframe(df_recs_display, use_container_width=True)
    else:
        st.success('Este cliente ya tiene un portafolio óptimo — No requiere cambios')
