# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pyspark.sql import SparkSession

spark=SparkSession.builder.appName('Practice').getOrCreate()
file_location = "salaries_clean.csv"
file_type = "csv"
df = spark.read.csv(file_location, header=True, inferSchema=True)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def mostrar():
    df.printSchema() #muestra la arquitectura de las columnas del excel
    df.show()# muestra el excel completo



#indexa una columna de string , es decir le da una equivalencia numerica
def indexar():
    from pyspark.ml.feature import StringIndexer
    global  df_r

    #indexer = StringIndexer(inputCol="location_state", outputCol="location_state_index",handleInvalid="skip")
    indexer = StringIndexer(inputCols=["location_state","location_country","job_title_category","job_title_rank","total_experience_years"],
                            outputCols=["location_state_indexed","location_country_indexed","job_title_category_indexed","job_title_rank_indexed","total_experience_years_indexed"],handleInvalid="skip")
    df_r=indexer.fit(df).transform(df)
    df_r.show()


def vector():
    from pyspark.ml.feature import VectorAssembler
    global finalized_data
    assembler=VectorAssembler(inputCols=["location_state_indexed","location_country_indexed","job_title_category_indexed","job_title_rank_indexed","total_experience_years_indexed"],
                    outputCol="Independent Features")
    output = assembler.transform(df_r)
    output.show()
    output.select("Independent Features").show()
    finalized_data= output.select("Independent Features","annual_base_pay")

#regresion lineal (prediccion en base a datos pasados)
#ejemplo de regresion lineal en el que se predice el salario mas alto
def modeloRegresion():
    from pyspark.ml.regression import LinearRegression

    finalized_data.show()
    train_data,test_data=finalized_data.randomSplit([0.75,0.25])
    regressor = LinearRegression(featuresCol="Independent Features", labelCol="annual_base_pay")
    regressor = regressor.fit(train_data)


    regressor.evaluate(test_data).predictions.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    mostrar()
    df.select("annual_base_pay").show()
    indexar()
    vector()
    modeloRegresion()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
