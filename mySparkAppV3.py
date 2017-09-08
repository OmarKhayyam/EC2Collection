from pyspark.sql import SparkSession,DataFrame,Column
from pyspark.sql import functions as Fun
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os,sys

# Use Amazon S3 for input data and output destination, 
# replace with your source and destination buckets. 
# This code expects to find the file with our personal 
# ratings in the source bucket, you can choose to add
# a prefix to the destination and source buckets.

srcBkt = "s3://myBucket/"
destBkt = "s3://myBucket/"
outfilename = "./mySparkApp.out"

if __name__ == "__main__":

# We change the default STDOUT for our program to
# point to a file in the local file system to ensure
# that we cleanup before using the program again.

	if os.path.exists(outfilename):
		os.remove(outfilename)

# Setting STDOUT to a new file

	orig_stdout = sys.stdout
	outfile = open(outfilename,"w")
	sys.stdout = outfile

# Build the Apache Spark session, this is the object
# through which we will be accessing all the capabilities
# that we need from Spark.

	spark = SparkSession.builder.appName("My Recommender System").getOrCreate()

# We get the ratings provided by the various users for
# the various movies and build our DataFrame.

	ratingsDF = spark.read.option("header","true").csv(srcBkt+"ratings.csv")
	seed = 1234

# We split the DataFrame to create two new DataFrames,
# the new DataFrames are for training and
# testing, they are created in the proportion 80% for
# training and 20% each for validation and testing.
# Note that while we have split the dataset in two,
# we can manage with two because Spark ML uses a k-fold
# cross validator (CV). The CV has some shortcomings that
# will be discussed below.

	(trngDF,testDF) = ratingsDF.randomSplit([0.8,0.2],seed)
	trngDF.persist()
	testDF.persist()

# If we have any personal preferences that we would 
# like to test we will find them in the personalRatings.txt
# file, we assume this file is in Amazon S3.
# We also set the datatypes right for all the datasets.

	myRatingsDF = spark.read.option("header","false").csv(srcBkt+"personalRatings.txt")
	myRatingsDF.persist()

# Adding our personal preferences to our training set so that we
# fit the model accordingly.

	nTrngDF = trngDF.drop('timestamp').union(myRatingsDF)

# Make sure the data types are interpreted correctly

	nTrngDF = nTrngDF.select(nTrngDF.userId.cast("int"),nTrngDF.movieId.cast("int"),nTrngDF.rating.cast("float"))
	nTrngDF.persist()
	trngDF.unpersist()
	nTestDF = testDF.select(testDF.userId.cast("int"),testDF.movieId.cast("int"),testDF.rating.cast("float"))
	nTestDF.persist()
	testDF.unpersist()

# Setting up the hyper-parameters, we set up the Rank for
# matrices of the user and item matrices, remember we are
# using Low Rank Matrix Factorization model. We are also
# setting the maximum number of iterations, our learning
# algorithm, in this case Alternating Least Squares (ALS),
# will make over the training set. We set the regularization
# parameter and the learning rate, both affect the quality
# of the model, and we may have to iterate multiple times
# to get these hyper-parameters just right, to get a model
# of aacceptable quality.

	irank = 25
	iterations = 20
	ireg = 0.2
	ilearn = 0.1
	model = None

# Iterate over the possibilities, of course, there could be
# more combinations of the above hyper parameters, but for 
# this example, we are examining only the above values.
# We then fit the model to the data in our training DataFrame.
# We instantiate an evaluator to evaluate our model using
# Root Mean Squared Error (RMSE) as the evaluation metric.

	alsresult = ALS(rank=irank,maxIter=iterations,regParam=ireg,alpha=ilearn,userCol="userId",itemCol="movieId",ratingCol="rating")
	model = alsresult.fit(nTrngDF)
	evaluator = RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")

# We get the predictions using the transformer by passing
# the test DataFrame, we then evaluate the quality of the model.
# We skip the 'nan' values due to a limitation of the Apache
# Spark CV. You can read all about it at the following
# URL : https://issues.apache.org/jira/browse/SPARK-14489

	predictions = model.transform(nTestDF)
	newrmse = evaluator.evaluate(predictions.filter(predictions.prediction != float('nan')))

# We now compute the RMSE for a model where we assume 
# ratings are an average of the ratings provided in the
# input data. We need to compare this to ensure that our
# model outperforms, at the least, a simplified model such as
# this.

	trng_avg_rating = nTrngDF.agg(Fun.avg(Fun.col("rating"))).alias("avg").collect()
# Adding a column to test set to introduce average prediction column
	nAvgTestDF = nTestDF.withColumn("prediction",Fun.lit(trng_avg_rating[0][0]))
	nTestDF.unpersist()

# Finally, we compute the ratings and thus the top twenty movies
# that the user has not seen previously which our model predicts
# the user 0 will like in descending order.

	print("************************************")
	print("*Top 20 recommended movies for you:*")
	print("************************************")
# Find movies that I have not rated
	allMoviesDF = spark.read.option("header","true").csv(srcBkt+"movies.csv")
	nallMoviesDF = (allMoviesDF.drop('genres').drop('title'))
	nmyRatingsDF = myRatingsDF.drop('_c0').drop('_c2').drop('_c3')
	nmyRatingsDF = nmyRatingsDF.select(nmyRatingsDF._c1.cast("int"))
	myRatingsDF.unpersist()
	nnallMoviesDF = nallMoviesDF.select(nallMoviesDF.movieId.cast("int"))
	unratedMoviesDF = nnallMoviesDF.subtract(nmyRatingsDF)
	unratedMoviesDF = unratedMoviesDF.withColumn("userId",Fun.lit(0))
# Lets get the predictions
	predictedRatingsDF = model.transform(unratedMoviesDF)
	predictedRatingsDF = (predictedRatingsDF.filter(predictedRatingsDF['prediction'] != float('nan'))).orderBy("prediction",ascending=False)
# Get the top 20 recommended movies for us
	for data in predictedRatingsDF.take(20):
		print("{}".format((allMoviesDF.select("title").filter(data.movieId == allMoviesDF.movieId).collect())[0]['title']))

	sys.stdout = orig_stdout
	outfile.close()

# We store the recommendations in our Amazon S3 bucket. Cleanup
# after us and stop the Spark Session.

	os.system("aws s3 cp "+outfilename+" "+destBkt)

# Hoping we have transferred the results to S3
	if os.path.exists(outfilename):
		os.remove(outfilename)
# Stopping the Spark session
	spark.stop()
