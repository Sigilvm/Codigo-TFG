### Los archivos de la forma algoritmo_script.py son versiones iniciales en las que los parámetros de entrada se definen como variables globales. Se ejecutan como scripts.

## En "methods.py" se adaptan los scripts a formato de función, para facilitar los tests. En este fichero están todos los scripts adaptados y cada uno cuenta además con dos versiones: una secuencial y una paralela. La diferencia entre ellas reside únicamente en la llamada al parámetro parallel=true en el decorador jit que precede a ciertas funciones.

Los archivos de la forma algoritmo_granularidad.py contienen versiones en formato función que reciben como parámetros los mismos que las funciones en "methods.py" pero añandiendo dos parámetros relativos a la granularidad, la cantidad de subdivisiones y el intervalo de comunicación. 

El archivo "funciones.py" incluye definiciones de funciones que se usarán para los tests, además de agrupar cada una de ellas junto a información sobre su óptimo global, su dimensión, el rango de búsqueda habitual... 
# Todos estos archivos (salvo los de formato script) se importan en "testing.py". En ese código se encuentran funciones dedicadas al testeo de resultados, con indicaciones sobre su uso en formato de comentario en el propio código.
