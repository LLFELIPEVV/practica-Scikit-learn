from scipy import constants

# Scipy esta enfocado en implementaciones cientificas, por lo que proporciona muchas constantes cientificas integradas.
# Ejemplo: usar la constante de Pi.
print(constants.pi)

# Unidades constantes
# Se puede ver una lista de todas las unidades bajo el modulo de constantes usando la funcion dir().
print(dir(constants))

# Categorias de unidades
# Las unidades se clasifican en estas categorias:
# Metric
# Binary
# Mass
# Angle
# Time
# Length
# Pressure
# Volume
# Speed
# Temperature
# Energy
# Power
# Force

# Prefijos Metricos (SI)
# Devuelve la unidad en metros.
print(constants.yotta)
print(constants.zetta)
print(constants.exa)
print(constants.peta)
print(constants.tera)
print(constants.giga)
print(constants.mega)
print(constants.kilo)
print(constants.hecto)
print(constants.deka)
print(constants.deci)
print(constants.centi)
print(constants.milli)
print(constants.micro)
print(constants.nano)
print(constants.pico)
print(constants.femto)
print(constants.atto)
print(constants.zepto)

# Prefijos Binarios
# Devuelve la unidad en bytes.
print(constants.kibi)
print(constants.mebi)
print(constants.gibi)
print(constants.tebi)
print(constants.pebi)
print(constants.exbi)
print(constants.zebi)
print(constants.yobi)

# Masa
# Devuelve la unidad en kilogramos.
print(constants.gram)
print(constants.metric_ton)
print(constants.grain)
print(constants.lb)
print(constants.pound)
print(constants.oz)
print(constants.ounce)
print(constants.stone)
print(constants.long_ton)
print(constants.short_ton)
print(constants.troy_ounce)
print(constants.troy_pound)
print(constants.carat)
print(constants.atomic_mass)
print(constants.m_u)
print(constants.u)
