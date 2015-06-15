package smodelkit;

import static java.lang.System.out;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.io.FilenameUtils;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;

import smodelkit.evaluator.AccuracyOfGroup;
import smodelkit.evaluator.AccuracyPerColumn;
import smodelkit.evaluator.Evaluator;
import smodelkit.evaluator.MSE;
import smodelkit.evaluator.TopN;
import smodelkit.evaluator.TopNHamming;
import smodelkit.filter.Filter;
import smodelkit.filter.MeanModeUnknownFiller;
import smodelkit.filter.NominalToCategorical;
import smodelkit.filter.Normalize;
import smodelkit.filter.ReorderOutputs;
import smodelkit.learner.HMONN;
import smodelkit.learner.IndependentClassifiers;
import smodelkit.learner.KNN;
import smodelkit.learner.MaxWeightEnsemble;
import smodelkit.learner.MonolithicTransformation;
import smodelkit.learner.NeuralNet;
import smodelkit.learner.NeuralNetAparapi;
import smodelkit.learner.NeuralNetCL;
import smodelkit.learner.OneClassWrapper;
import smodelkit.learner.RankedCC;
import smodelkit.learner.SupervisedLearner;
import smodelkit.learner.VotingEnsemble;
import smodelkit.learner.WOVEnsemble;
import smodelkit.learner.ZeroR;
import smodelkit.util.Counter;
import smodelkit.util.Helper;
import smodelkit.util.Logger;
import smodelkit.util.Pair;
import smodelkit.util.Plotter;
import smodelkit.util.Range;
import smodelkit.util.SerializationUtilities;
import smodelkit.util.TestRandom;
import smodelkit.util.ThreadCounter;
import smodelkit.util.Tuple3;

/**
 * For training and evaluating learners.
 */
public class MLSystemsManager
{			
	/**
	 * If data is not null, it will be used as the data. If static evaluation is used, data will be
	 * the training set. Unknown filling and column removal are still applied.
	 * @return A map from evaluator types to the test results with that evaluator.
	 * @throws IOException 
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 * @throws ClassNotFoundException 
	 */
	public Evaluation run(String[] args, Matrix data)
			throws IOException, InterruptedException, ExecutionException, ClassNotFoundException
	{	
		if (data != null)
		{
			// This is a hack to make sure I always print the args when doing a uci run.
			out.print("MLSystemsManager called with args: ");
			for (String arg : args)
			{
				System.out.print(arg + " ");
			}
			System.out.println();
		}
		else
		{
			Logger.print("MLSystemsManager called with args: ");
			for (String arg : args)
			{
				Logger.print(arg + " ");
			}
			Logger.println();
		}
		
		// Parse the command line arguments
		ArgParser parser = ArgParser.parse(args);
		
		if (parser.maxThreads != null)
			ThreadCounter.setMaxThreads(parser.maxThreads);
		
		Matrix.useDouble = !parser.useFloats;
				
		Random rand = null;
		if (parser.seedStr == null || parser.seedStr.equals(""))
		{
			rand = new Random();
		}
		else if (parser.seedStr != null && parser.seedStr.equals("test"))
		{
			rand = new TestRandom();
		}
		else
		{
			Integer randSeed = Integer.parseInt(parser.seedStr);
			rand = new Random(randSeed);
		}
		
		if (data == null)
		{
			data = new Matrix();
			if (parser.dataset.size() == 1)
			{
				// Load the ARFF file
				String fileName = parser.dataset.get(0);

				if (!fileName.endsWith(".arff"))
					throw new IllegalArgumentException("When only 1 parameter is given with -A, it must be an" +
							" arff file name. -A arguments were: " + parser.dataset);
				data.loadFromArffFile(fileName, true, parser.maxRows);		
			}
			else
			{
				if (parser.dataset.size() != 2)
					throw new IllegalArgumentException("Unsupported number of parameters given to -A.");
				String namesFilename = parser.dataset.get(0);
				String dataFilename = parser.dataset.get(1);
				if (!namesFilename.endsWith(".names"))
					throw new IllegalArgumentException("When 2 parameters are given with -A, the first parameter" +
							" must be a .names file.");
				if (!dataFilename.endsWith(".data") && dataFilename.endsWith(".test"))
					throw new IllegalArgumentException("When 2 parameters are given with -A, the second parameter" +
							" must be a .data or .test file.");
				data.loadFromNamesFormat(namesFilename, dataFilename);
			}
		}
		
		deleteColumns(parser.ignoredColumns, data);
										
		if (parser.numLabelColumns != null && parser.labelColumnNames != null 
				&& parser.numLabelColumns != parser.labelColumnNames.size())
			throw new IllegalArgumentException("Number of label indexes must match the number of label" +
					" columns specified (if specified).");
				
		if (parser.numLabelColumns != null)
		{
			// Override any existing value for numLabelColumns in data.
			data.setNumLabelColumns(parser.numLabelColumns);
		}
		else if (parser.labelColumnNames != null)
		{
			data.setNumLabelColumns(parser.labelColumnNames.size());
			data = parseAndMoveLabelColumns(data, parser.labelColumnNames);
		}

		MetadataPrinter.printMetadata(data);
		MetadataPrinter.printMetadataPerAttribute(data);
		
		if (parser.printMetadataOnly)
			return null;

		if (parser.learner != null)
		{
			String learnerName = parser.learner.get(0);
			Logger.println("Learning algorithm: " + learnerName);
		}

		data = fillOrRemoveUnknownData(parser.unknownFiller, data);
		
		if (parser.oversample)
		{
			if (parser.labelColumnNames != null || parser.numLabelColumns != null 
					|| (parser.numLabelColumns != null && parser.numLabelColumns > 1))
			{
				throw new IllegalArgumentException("To oversample you must specify only 1 label column.");
			}
		}


		Logger.println();
		String evalMethod = parser.evaluation.get(0);
		Logger.println("Evaluation method: " + evalMethod);
		List<String> evalParameters = parser.evaluation.subList(1, parser.evaluation.size()); 
		Logger.println();

		// The Normalize filter needs the entire dataset when it is initialized.
		if (evalMethod.equals("training"))
		{
			if (parser.oversample)
			{
				Logger.println("Oversampling training set.");
				Logger.println("Rows before: " + data.rows());
				data = data.oversample(rand);
				Logger.println("Rows after: " + data.rows());
			}
			SupervisedLearner learner = getLearner(rand, parser);
			Logger.println("Evaluations will be on the training set...");
			Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
			data = null;
			Matrix inputs = inputsAndLabels.getFirst();
			Matrix labels = inputsAndLabels.getSecond();
			labels = reorderLabelColumns(labels, parser.labelColumnOrder);
			List<Evaluator> testEvaluators = 
					getTestEvaluators(parser.evaluators, labels);
			
			if (parser.printPercentUniqueTestLabels)
			{
				// This should always be zero.
				Logger.println("Percent unique labels in test set: " + findPercentUniqueTestLabels(labels, labels));
			}
			
			if (parser.deserializeFileName == null)
			{
				double startTime = System.currentTimeMillis();
				learner.train(inputs, labels);
				double elapsedTime = System.currentTimeMillis() - startTime;
				Logger.println("Time to train (in seconds): " + elapsedTime
						/ 1000.0);
				serializeLearner(learner);
			}
			double startTime = System.currentTimeMillis();
			Evaluation evaluation = Evaluator.runEvaluators(inputs, labels, learner, true, testEvaluators);
			Logger.println("Time to test (in seconds): " + (System.currentTimeMillis() - startTime)
					/ 1000.0);
			for (Evaluator evaluator : testEvaluators)
			{
				Logger.println("\nEvaluator: " + evaluator.getClass().getSimpleName());
				List<ConfusionMatrix> confusions = evaluation.getConfusions(evaluator.getClass());
				if (confusions != null)
				{
					printConfusionMatrices(confusions);
				}
				List<Double> scores = evaluation.getScores(evaluator.getClass());
				Logger.println("Training set evaluation: " + (scores.size() == 1 ? scores.get(0) : scores));
			}
			return evaluation;
		} 
		else if (evalMethod.equals("static"))
		{
			if (parser.oversample)
			{
				Logger.println("Oversampling training set.");
				Logger.println("Rows before: " + data.rows());
				data = data.oversample(rand);
				Logger.println("Rows after: " + data.rows());
			}

			SupervisedLearner learner = getLearner(rand, parser);
			
			Pair<Matrix> pair = data.splitInputsAndLabels();
			Matrix inputs = pair.getFirst();
			Matrix labels = pair.getSecond();
			data = null;
			pair = null;
			labels = reorderLabelColumns(labels, parser.labelColumnOrder);
						
			if (parser.deserializeFileName == null)
			{
				double startTime = System.currentTimeMillis();
				learner.train(inputs, labels);
				double elapsedTime = System.currentTimeMillis() - startTime;
				Logger.println("Time to train (in seconds): " + elapsedTime
						/ 1000.0);
				serializeLearner(learner);
			}
			
			List<Evaluator> testEvaluators = 
					getTestEvaluators(parser.evaluators, labels);

			double startTime = System.currentTimeMillis();
			Evaluation trainEvaluation = parser.includeTrainingDataEvaluations ? 
					Evaluator.runEvaluators(inputs, labels, learner, true, testEvaluators) : null;
			inputs = null;
			labels = null;
				
			Matrix testInputs, testLabels; 
			{
				Matrix testData = new Matrix();
				String filename = evalParameters.get(0);
				if (filename.endsWith(".arff"))
				{
					testData.loadFromArffFile(evalParameters.get(0));				
				}
				else if (filename.endsWith(".data") || filename.endsWith(".test"))
				{
					String namesFilename = parser.dataset.get(0);
					if (!namesFilename.endsWith(".names"))
						throw new IllegalArgumentException("Cannot mix .name format and .arff format for training and" +
								" test datasets.");
					testData.loadFromNamesFormat(namesFilename, filename);
				}
				else
				{
					throw new IllegalArgumentException("Unknown dataset file type: " + filename);
				}
				deleteColumns(parser.ignoredColumns, testData);
				testData = fillOrRemoveUnknownData(parser.unknownFiller, testData);
				Pair<Matrix> pairTest = testData.splitInputsAndLabels();
				testInputs = pairTest.getFirst();
				testLabels = pairTest.getSecond();
				testLabels = reorderLabelColumns(testLabels, parser.labelColumnOrder);
			}
			
			if (parser.printPercentUniqueTestLabels)
			{
				// This should always be zero.
				Logger.println("Percent unique labels in test set: " + findPercentUniqueTestLabels(labels, testLabels));
			}
			
			Logger.println("Evaluations will be on a separate test set...");
			Logger.println("Test set name: " + evalParameters.get(0));
			Logger.println("Number of test instances: " + testInputs.rows());

			Evaluation testEvaluation = Evaluator.runEvaluators(testInputs, testLabels, learner, true, testEvaluators);
			Logger.println("Time to test (in seconds): " + (System.currentTimeMillis() - startTime)
					/ 1000.0);
			for (Evaluator evaluator : testEvaluators)
			{
				Logger.println("\nEvaluator: " + evaluator.getClass().getSimpleName());
				if (parser.includeTrainingDataEvaluations)
				{
					List<Double> trainAccuracy = trainEvaluation.getScores(evaluator.getClass());
					Logger.println("Training set evaluation: " + (trainAccuracy.size() == 1 ? trainAccuracy.get(0) : trainAccuracy));
				}
				List<ConfusionMatrix> confusions = testEvaluation.getConfusions(evaluator.getClass());
				if (confusions != null)
				{
					printConfusionMatrices(confusions);
				}
				List<Double> testAccuracy = testEvaluation.getScores(evaluator.getClass());
				Logger.println("Test set evaluation: " + (testAccuracy.size() == 1 ? testAccuracy.get(0) : testAccuracy));
			
			}
			return testEvaluation;
		} 
		else if (evalMethod.equals("random"))
		{
			SupervisedLearner learner = getLearner(rand, parser);
			Logger.println("Evaluations will be on a random hold-out set...");
			double trainPercent = 1 - Double.parseDouble(evalParameters.get(0));
			if (trainPercent < 0 || trainPercent > 1)
				throw new IllegalArgumentException(
						"Percentage for random evaluation must be between 0 and 1");
			Logger.println("Percentage used for training: " + trainPercent);
			Logger.println("Percentage used for testing: "
					+ Double.parseDouble(evalParameters.get(0)));
			data.shuffle(rand);
			int trainSize = (int) (trainPercent * data.rows());
			Matrix trainInstances = new Matrix(data, 0, 0, trainSize, data.cols());
			if (parser.oversample)
			{
				Logger.println("Oversampling training set.");
				Logger.println("Rows before: " + trainInstances.rows());
				trainInstances = trainInstances.oversample(rand);
				Logger.println("Rows after: " + trainInstances.rows());
			}			
			Matrix trainInputs = new Matrix(trainInstances, 0, 0, trainInstances.rows(), data.cols() - data.getNumLabelColumns());
			Matrix trainLabels = new Matrix(trainInstances, 0, data.cols() - data.getNumLabelColumns(), trainInstances.rows(), data.getNumLabelColumns());
			trainLabels = reorderLabelColumns(trainLabels, parser.labelColumnOrder);
			trainInstances = null;
			Matrix testInputs = new Matrix(data, trainSize, 0, data.rows() - trainSize, 
					data.cols() - data.getNumLabelColumns());
			Matrix testLabels = new Matrix(data, trainSize, data.cols() - data.getNumLabelColumns(),
					data.rows() - trainSize, data.getNumLabelColumns());
			data = null;
			testLabels = reorderLabelColumns(testLabels, parser.labelColumnOrder);
			
			if (parser.printPercentUniqueTestLabels)
			{
				// This should always be zero.
				Logger.println("Percent unique labels in test set: " + findPercentUniqueTestLabels(trainLabels, testLabels));
			}			
			
			List<Evaluator> testEvaluators = 
					getTestEvaluators(parser.evaluators, trainLabels);

			if (parser.deserializeFileName != null)
				throw new IllegalArgumentException("It doesn't make much sense to serialize a model and reuse it on random" +
						" training and test sets.");
			
			Logger.println("Training the learner.");
			double startTime = System.currentTimeMillis();
			learner.train(trainInputs, trainLabels);
			double elapsedTime = System.currentTimeMillis() - startTime;
			Logger.println("Time to train (in seconds): " + elapsedTime
					/ 1000.0);
				
			startTime = System.currentTimeMillis();
			Evaluation trainEvaluation = parser.includeTrainingDataEvaluations ? 
					Evaluator.runEvaluators(trainInputs, trainLabels, learner, true, testEvaluators) : null;
			Evaluation testEvaluation = Evaluator.runEvaluators(testInputs, testLabels, learner, true, testEvaluators);
			Logger.println("Time to test (in seconds): " + (System.currentTimeMillis() - startTime)
					/ 1000.0);
			for (Evaluator evaluator : testEvaluators)
			{
				Logger.println("\nEvaluator: " + evaluator.getClass().getSimpleName());
				if (parser.includeTrainingDataEvaluations)
				{
					List<Double> trainAccuracy = trainEvaluation.getScores(evaluator.getClass());
					Logger.println("Training set evaluation: " + (trainAccuracy.size() == 1 ? trainAccuracy.get(0) : trainAccuracy));
				}
				List<ConfusionMatrix> confusions = testEvaluation.getConfusions(evaluator.getClass());
				if (confusions != null)
				{
					printConfusionMatrices(confusions);
				}
				List<Double> testAccuracy = testEvaluation.getScores(evaluator.getClass());
				Logger.println("Test set evaluation: " + (testAccuracy.size() == 1 ? testAccuracy.get(0) : testAccuracy));
			
			}
			return testEvaluation;

		} 
		else if (evalMethod.equals("cross"))
		{
			if (parser.deserializeFileName != null)
				throw new IllegalArgumentException("Cross validation does not support loading serialized models.");
			Logger.println("Evaluations will use cross-validation...");
			
			final int folds = Integer.parseInt(evalParameters.get(0));
			if (folds <= 0)
				throw new IllegalArgumentException("Number of folds must be greater than 0");
			
			final int reps = evalParameters.size() > 1 ? Integer.parseInt(evalParameters.get(1)) : 1;	
			if (reps <= 0)
				throw new IllegalArgumentException("Number of reps must be greater than 0");
			if (evalParameters.size() > 2)
			{
				throw new IllegalArgumentException(String.format("Expected at most 2 evaluation parameters, but received %d.", evalParameters.size()));
			}
			
			int threadsReserved = ThreadCounter.reserveThreadCount(folds*reps);
			int numThreads = Math.max(1, threadsReserved);
			// Only do this if --threads was not used.
			if (folds % numThreads != 0 && numThreads < folds && !ThreadCounter.isMaxThreadsSetByUser())
			{
				// increase numThreads until it evenly divides the number of folds.
				while(folds % numThreads != 0)
					numThreads++;
			}

			Logger.println("Number of folds: " + folds);
			Logger.println("Number of reps: " + reps);
			
			// For holding test set evaluations from each fold.
			List<Evaluation> evaluationsPerFold = new ArrayList<>();
			List<Double> timesPerFold = new ArrayList<>();
			
			// Each fold runs in a separate thread (up to the number of threads allowed).
			// Note: When using a fixed random seed, results will still vary if more than one thread is used because
			// the threads all share the same random number generator.
			double startTime = System.currentTimeMillis();
			ExecutorService exService = Executors.newFixedThreadPool(numThreads);
			try 
			{
				List<Future<Tuple3<Evaluation, Evaluation, Double>>> futures = new ArrayList<>();
				for (int j = 0; j < reps; j++)
				{
					data.shuffle(rand);
					for (int i = 0; i < folds; i++)
					{						
						int foldSize = data.rows() / folds;
						int begin = i * foldSize;
						int end = (i + 1) * foldSize; // data.rows() / folds = foldSize
						Matrix trainInstances = new Matrix(data, 0, 0, begin, data.cols());
						trainInstances.add(data, end, 0, data.rows() - end);
						if (parser.oversample)
						{
							Logger.println("Oversampling training set.");
							Logger.println("Rows before: " + trainInstances.rows());
							trainInstances = trainInstances.oversample(rand);
							Logger.println("Rows after: " + trainInstances.rows());
						}			
						final Matrix trainInputs = new Matrix(trainInstances, 0, 0, trainInstances.rows(),
								trainInstances.cols() - data.getNumLabelColumns());
						Matrix trainLabelsTemp = new Matrix(trainInstances, 0, trainInstances.cols() - 
								data.getNumLabelColumns(), trainInstances.rows(), data.getNumLabelColumns());
						final Matrix trainLabels = reorderLabelColumns(trainLabelsTemp, parser.labelColumnOrder);
						trainLabelsTemp = null;
						trainInstances = null;

						final Matrix testInputs = new Matrix(data, begin, 0, end - begin, data.cols() - data.getNumLabelColumns());
						final Matrix testLabels = reorderLabelColumns(new Matrix(data, begin,
								data.cols() - data.getNumLabelColumns(), end - begin, data.getNumLabelColumns()), parser.labelColumnOrder);
						
						if (testLabels.rows() == 0)
							throw new IllegalArgumentException("The dataset is not large enough for the number of" +
									" folds specified.");
						
						if (parser.printPercentUniqueTestLabels)
						{
							// This should always be zero.
							Logger.println("Percent unique labels in test set for rep " + j + " fold " + i + ": "
									+ findPercentUniqueTestLabels(trainLabels, testLabels));
						}			
						
						// Create a new Random so that I don't access it across multiple threads at once.
						final SupervisedLearner learner = getLearner(new Random(rand.nextLong()), parser);
						
						final List<Evaluator> testEvaluators = getTestEvaluators(parser.evaluators, trainLabels);

						Callable<Tuple3<Evaluation, Evaluation, Double>> callable = 
								new Callable<Tuple3<Evaluation, Evaluation, Double>>()
						{						
							@Override
							public Tuple3<Evaluation, Evaluation, Double> call() throws Exception
							{
								long startTime = System.currentTimeMillis();
								learner.train(trainInputs, trainLabels);
								Evaluation trainEval;
								if (parser.includeTrainingDataEvaluations)
								{
									trainEval = Evaluator.runEvaluators(trainInputs, trainLabels, learner,
										true, testEvaluators);
								}
								else
								{
									trainEval = new Evaluation();
								}
								Evaluation testEval = Evaluator.runEvaluators(testInputs, testLabels, learner,
										true, testEvaluators);
								double elapsedTimeInSeconds = (System.currentTimeMillis() - startTime)/1000.0;								
								return new Tuple3<>(trainEval, testEval, elapsedTimeInSeconds);
							}
						};
						futures.add(exService.submit(callable));
					}					
				}
				// Retrieve the results from each run.
				for (int repI = 0; repI < reps; repI++)
				{
					for (int foldI = 0; foldI < folds; foldI++)
					{
						int index = folds*repI + foldI;
						Tuple3<Evaluation, Evaluation, Double> foldResult = futures.get(index).get();
						evaluationsPerFold.add(foldResult.getSecond());
						timesPerFold.add(foldResult.getThird());
						Logger.println("Rep=" + repI + ", Fold=" + foldI);
						for (Class<? extends Evaluator> evalType : foldResult.getSecond().getEvaluatorTypes())
						{
							Logger.println(evalType.getSimpleName() + " test: " 
									+ Helper.formatDoubleList(foldResult.getSecond().getScores(evalType)) 
									+ (parser.includeTrainingDataEvaluations ? ", " + "training: " 
									+ Helper.formatDoubleList(foldResult.getFirst().getScores(evalType)) : ""));	
						}
						Logger.println("Time to train and test (in seconds): " + Helper.formatDouble(foldResult.getThird())); 
					}
				}

			}
			finally
			{
				exService.shutdown();
				ThreadCounter.freeThreadCount(threadsReserved);
			}
			double elapsedTime = System.currentTimeMillis() - startTime;
			Logger.println("Elapsed time total (in seconds): " + elapsedTime / 1000.0);
			double averageTime = timesPerFold.stream().mapToDouble(d -> d).average().getAsDouble();
            Logger.print("Average time per learner (in seconds): " + averageTime);

			// sumResults maps evaluator types to summed test set results.
			Evaluation sumResults = new Evaluation();
			for (Evaluation eval : evaluationsPerFold)
			{
				for (Class<? extends Evaluator> evalType : evaluationsPerFold.get(0).getEvaluatorTypes())
				{
					if (!sumResults.getEvaluatorTypes().contains(evalType))
					{
						sumResults.putScores(evalType, new ArrayList<>(eval.getScores(evalType)));
					}
					else
					{
						// Add in the new values.
						for (int i = 0; i < eval.getScores(evalType).size(); i++)
							sumResults.getScores(evalType).set(i, sumResults.getScores(evalType).get(i) 
									+ eval.getScores(evalType).get(i));
					}
				}
			}
			Logger.println(", standard deviation: " + Helper.formatDouble(Helper.findStandardDeviation(timesPerFold)));
			
			Evaluation standardDeviations = new Evaluation();
			for (Class<? extends Evaluator> evalType : sumResults.getEvaluatorTypes())
			{
				// For all evaluations of type evalType, find the standard deviation for each
				// dimension of the evaluation.
				List<Double> standardDevsForEvalType = new ArrayList<>();
				for (int i : new Range(evaluationsPerFold.get(0).getScores(evalType).size()))
				{
					// Get all of the scores for the i'th dimension of the evaluations of type evalType.
					List<Double> scores = evaluationsPerFold.stream().map(
							eval -> eval.getScores(evalType).get(i)).collect(Collectors.toList());
					double standardDev = Helper.findStandardDeviation(scores);
					standardDevsForEvalType.add(standardDev);
				}
				standardDeviations.putScores(evalType, standardDevsForEvalType);
			}


			Evaluation meanResults = new Evaluation();
			for (Class<? extends Evaluator> evalType : sumResults.getEvaluatorTypes())
			{
				meanResults.putScores(evalType, Helper.map(sumResults.getScores(evalType), 
						new Function<Double, Double>()
					{
						@Override
						public Double apply(Double item)
						{	
							return item / (reps * folds);
						}
					}));
				
			}
			Logger.print("Mean evaluations: ");
			Logger.println(meanResults);
			Logger.print("Standard deviations: " );
			Logger.println(standardDeviations);
			return meanResults;
		}
		throw new IllegalStateException(String.format("Unknown evalutaion method \"%s\"", parser.evaluation.get(0)));
	}

	public static double findPercentUniqueTestLabels(Matrix trainLabels, Matrix testLabels)
	{
		// Note that Vectors with different weights will be considered unique.
		Set<Vector> trainSet = new TreeSet<>();
		trainLabels.stream().forEach(l -> trainSet.add(l));
		long numUnique = testLabels.stream().filter(l -> !trainSet.contains(l)).count();
		return (double)numUnique/testLabels.rows();
	}

	/**
	 * Moves the specified label column names to the end of the matrix to be used as labels.
	 * @param dataset
	 * @param labelNames 
	 * @return
	 */
	private static Matrix parseAndMoveLabelColumns(final Matrix dataset, List<String> labelNames)
	{	
		List<Integer> indexes = new ArrayList<Integer>();
		for (String lName : labelNames)
		{
			boolean found = false;
			for (int c = 0; c < dataset.cols(); c++)
			{
				if (dataset.getAttrName(c).equals(lName))
				{
					indexes.add(c);
					found = true;
					break;
				}
			}
			if (!found)
				throw new IllegalArgumentException(String.format("The dataset has no column named" +
				" \"%s\"", lName));
		}
		return moveLabelColumns(dataset, indexes);
	}
	
	public static Matrix moveLabelColumns(final Matrix dataset, List<Integer> indexes)
	{		
		indexes = new ArrayList<>(indexes);
		final Matrix result = new Matrix(dataset);

		// Change negative attributes to be offset from the end of the list.
		for (int i = 0; i < indexes.size(); i++)
		{
			if (indexes.get(i) < 0)
				indexes.set(i, dataset.cols() - Math.abs(indexes.get(i)));
		}

		// Take note of which columns should be moved.
		List<String> expectedLabelAttrNames = new ArrayList<String>();
		for (int i : indexes)
		{
			expectedLabelAttrNames.add(result.getAttrName(i));
		}
		int resultSize = result.cols();
					
		// Remove the columns that will be moved to the labels.
		Matrix movedCols = new Matrix();
		for (int i = 0; i < indexes.size(); i++)
		{
			movedCols.copyColumns(result, indexes.get(i), 1);
			result.removeColumn(indexes.get(i));
			// Decrement the remaining indexes.
			for (int j = i + 1; j < indexes.size(); j++)
			{
				// Shouldn't have 2 of the same index.
				if (indexes.get(j) == indexes.get(i))
					throw new IllegalArgumentException();
				
				if (indexes.get(j) > indexes.get(i))
					indexes.set(j, indexes.get(j) - 1);
			}
		}
				
		// Put all of the labels back in at the end.
		result.copyColumns(movedCols, 0, movedCols.cols());
		
		// Make sure we have the right label columns.
		for (int i = 0; i < expectedLabelAttrNames.size(); i++)
		{
			int resultIndex = result.cols() - indexes.size() + i;
			if (!result.getAttrName(resultIndex).equals(expectedLabelAttrNames.get(i)))
				throw new IllegalStateException(String.format("Expected %s but got %s for column %d. indexes: %s",
						expectedLabelAttrNames.get(i), result.getAttrName(resultIndex), resultIndex,
						indexes));
		}
		
		// Make sure the result is as expected.
		if (result.cols() != resultSize)
			throw new IllegalStateException("Result is the wrong size.");
										
		return result;

	}
	
	private static void printConfusionMatrices(List<ConfusionMatrix> confusions)
	{
		if (confusions.size() == 0)
			return;
		Logger.println();
		for (ConfusionMatrix m : confusions)
		{
			Logger.format("Confusion matrix for \"%s\": (Row=target value, Col=predicted value)\n",
					m.getLabelName());
			Logger.println(m);
			Logger.println("Attribute evaluations:");
			Logger.print(m.getAccuracyPerAttributePrinted());
			Logger.println(m.getAccuracyPrinted());
		}
	}
	
	private static void serializeLearner(SupervisedLearner learner) throws IOException
	{
		double startTime = System.currentTimeMillis();
		if (!Files.exists(Paths.get("models")))
		{
			Files.createDirectory(Paths.get("models"));
		}
		SerializationUtilities.serialize(learner, 
				Paths.get("models", learner.getClass().getSimpleName() + ".ser").toString());
		double elapsedTime = System.currentTimeMillis() - startTime;
		Logger.println("Time to serialize learner (in seconds): " + elapsedTime
				/ 1000.0);
	}
	
	/**
	 * Either creates a new learner or deserializes one, depending on parser.
	 * @param rand
	 * @param parser
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	private static SupervisedLearner getLearner(Random rand, ArgParser parser) 
			throws IOException, ClassNotFoundException
	{
		if (parser.deserializeFileName != null)
		{
			Path path = Paths.get("models", parser.deserializeFileName);
			Logger.println("Loading learner from " + path.toString());
			return (SupervisedLearner) SerializationUtilities.deserialize(path.toString());
		}
		else
		{
			String learnerName = parser.learner.get(0);
			String configFileName = parser.learner.get(1);
			Plotter.setFilePrefix(FilenameUtils.getBaseName(configFileName) + "_");;
			JSONObject settings = parseModelSettingsFile(configFileName);
			return createLearner(rand, learnerName, settings);
		}
	}
	
	public static SupervisedLearner createLearner(Random rand, String learnerName, String configFileName) throws IOException
	{
		JSONObject settings = parseModelSettingsFile(configFileName);
		return createLearner(rand, learnerName, settings);
	}
	
	/**
	 * When you make a new learning algorithm, you should add a line for it to
	 * this method.
	 */
	@SuppressWarnings("unchecked")
	public static SupervisedLearner createLearner(Random rand, String learnerName, JSONObject settings)
	{
		SupervisedLearner learner = null;

		// Handle nicknames.
		if (learnerName.equals("neuralnet"))
		{
			learnerName = NeuralNet.class.getCanonicalName();
		}
		else if (learnerName.equals("neuralnet_cl"))
		{
			learnerName = NeuralNetCL.class.getCanonicalName();
		}
		else if (learnerName.equals("neuralnet_aparapi"))
		{
			learnerName = NeuralNetAparapi.class.getCanonicalName();
		}
		else if (learnerName.equals("rankedcc"))
		{
			learnerName = RankedCC.class.getCanonicalName();
		}
		else if (learnerName.equals("ic"))
		{
			learnerName = IndependentClassifiers.class.getCanonicalName();
		}
		else if (learnerName.equals("knn"))
		{
			learnerName = KNN.class.getCanonicalName();
		}
		else if (learnerName.equals("hmonn"))
		{
			learnerName = HMONN.class.getCanonicalName();
		}
		else if (learnerName.equals("voting_ensemble"))
		{
			learnerName = VotingEnsemble.class.getCanonicalName();
		}
		else if (learnerName.equals("mwe"))
		{
			learnerName = MaxWeightEnsemble.class.getCanonicalName();
		}		
		else if (learnerName.equals("wove"))
		{
			learnerName = WOVEnsemble.class.getCanonicalName();
		}
		else if (learnerName.equals("monolithic"))
		{
			learnerName = MonolithicTransformation.class.getCanonicalName();
		}
		else if (learnerName.equals("zeror"))
		{
			learnerName = ZeroR.class.getCanonicalName();
		}
		else if (learnerName.equals("weka"))
		{
			learnerName = "smodelkit.learner.WekaWrapper";
		}
		else if (learnerName.equals("one_class_wrapper"))
		{
			learnerName = OneClassWrapper.class.getCanonicalName();
		}
//		else if (learnerName.equals("pairwise_coupling"))
//		{
//			learnerName = PairwiseCoupling.class.getCanonicalName();
//		}
		
		Class<SupervisedLearner> learnerClass;
		try
		{
			learnerClass = (Class<SupervisedLearner>)Class.forName(learnerName);
			learner = learnerClass.newInstance();
		} 
		catch (ClassNotFoundException | InstantiationException | IllegalAccessException ex)
		{
			throw new RuntimeException(ex);
		}
		learner.setRandom(rand);
		learner.configure(settings);
		
		// Create the filters.
		Filter filter = createFilter(rand, settings);
		learner.setFilter(filter);

		
		return learner;
	}
	
	@SuppressWarnings("unchecked")
	private static Filter createFilter(Random rand, JSONObject settings)
	{
		Filter prevFilter = null;
		if (settings.containsKey("filters"))
		{
			JSONArray filterJSON = (JSONArray)settings.get("filters");
			List<String> filterNames = Arrays.asList(Helper.JSONArrayToStringArray(filterJSON));
			// Reverse the order of the filter names so that the filters will be applied in the order specified.
			Collections.reverse(filterNames);
			for (String filterName : filterNames)
			{
				String[] nameParts = filterName.split(" ");

				if (nameParts[0].equals("nom_to_categorical"))
				{
					nameParts[0] = NominalToCategorical.class.getCanonicalName();
				}
				else if (nameParts[0].equals("normalize"))
				{
					nameParts[0] = Normalize.class.getCanonicalName();
				}
				else if (nameParts[0].equals("reorder_outputs"))
				{
					nameParts[0] = ReorderOutputs.class.getCanonicalName();
				}
				else if (nameParts[0].equals("mean_mode_unknown_filler"))
				{
					nameParts[0] = MeanModeUnknownFiller.class.getCanonicalName();
				}
				
				Class<Filter> filterClass;
				Filter filter;
				try
				{
					filterClass = (Class<Filter>)Class.forName(nameParts[0]);
					filter = filterClass.newInstance();
				} 
				catch (ClassNotFoundException | InstantiationException | IllegalAccessException ex)
				{
					throw new RuntimeException(ex);
				}
				
				filter.setInnerFilter(prevFilter);
				prevFilter = filter;
				filter.setRandom(rand);
				filter.configure(Arrays.copyOfRange(nameParts, 1, nameParts.length));
				
			}
		}
		
		return prevFilter;
	}
		
	public static JSONObject parseModelSettingsFile(String fileName)
	{
		String contents;
		try
		{
			contents = Helper.readFile(fileName);
		} catch (IOException e)
		{
			throw new RuntimeException(e);
		}
		JSONObject result = (JSONObject)JSONValue.parse(contents);
		if (result == null)
			throw new IllegalArgumentException("Unable to parse settings file: " + fileName);
		return result;
	}
	
	@SuppressWarnings("unchecked")
	List<Evaluator> getTestEvaluators(List<String> args, Matrix labels)
	{
		List<Evaluator> result = new ArrayList<>();
		if (args == null)
		{
			// No evaluator was specified. Return a default one.
			if (labels.isContinuous(0))
				result.add(new MSE());
			else
			{
				TopN topN = new TopN();
				topN.configure(Arrays.asList(1));
				result.add(topN);
			}
			return result;
		}

		// Copy args to make sure we don't mutate it.
		args = new LinkedList<>(args);
		
		// Break up args using the delimiter "end".
		List<List<String>> argGroups = new ArrayList<>();
		argGroups.add(new ArrayList<>());
		for (String a : args)
		{
			if (a.equals("end"))
			{
				argGroups.add(new ArrayList<>());
			}
			else
			{
				argGroups.get(argGroups.size() - 1).add(a);
			}
		}
		args = null;
		
		for (List<String> group : argGroups)
		{
			Evaluator evaluator;
			if (group.get(0).toLowerCase().equals("top-n"))
			{
				evaluator  = new TopN();
			}
			else if (group.get(0).toLowerCase().equals("top-n-hamming"))
			{
				evaluator = new TopNHamming();
			}
			else if (group.get(0).toLowerCase().equals("percolumn"))
			{
				evaluator = new AccuracyPerColumn();
			}		
			else if (group.get(0).toLowerCase().equals("accuracy-group"))
			{
				evaluator = new AccuracyOfGroup();
			}
			else
			{
				Class<Evaluator> evaluatorClass;
				try
				{
					evaluatorClass = (Class<Evaluator>)Class.forName(group.get(0));
					evaluator = evaluatorClass.newInstance();
				} 
				catch (ClassNotFoundException | InstantiationException | IllegalAccessException ex)
				{
					throw new RuntimeException(ex);
				}
			}
			
			evaluator.configure(group.subList(1, group.size()).toArray(new String[group.size() - 1]));
			result.add(evaluator);
		}
		if (result.isEmpty())
			throw new IllegalStateException();
		return result;
	}
	
	void deleteColumnsByIndex(int[] ignoredAttributes, Matrix dataset)
	{
		// Delete unwanted data
		// Change negative attributes to be offset from the end of the list.
		for (int i = 0; i < ignoredAttributes.length; i++)
		{
			if (ignoredAttributes[i] < 0)
				ignoredAttributes[i] = dataset.cols() - Math.abs(ignoredAttributes[i]);
		}
		for (int i = 0; i < ignoredAttributes.length; i++)
		{
			//if (ignoredAttributes[i] >= dataset.cols() -1) // Make sure we don't delete the labels or go out of range.
			if ((int)ignoredAttributes[i] >= dataset.cols())
				throw new IllegalArgumentException("Ignore column too large");
			dataset.removeColumn(ignoredAttributes[i]);
			for (int j = i + 1; j < ignoredAttributes.length; j++)
				if (ignoredAttributes[j] > ignoredAttributes[i])
					ignoredAttributes[j]--;
		}
	}
	
	void deleteColumnsByName(List<String> ignoredAttributes, Matrix dataset)
	{
		for(String attr : ignoredAttributes)
		{
			boolean found = false;
			for (int i = 0; i < dataset.cols(); i++)
			{
				if (dataset.getAttrName(i).equals(attr))
				{
					dataset.removeColumn(i);
					found = true;
				}
			}
			if (!found)
				throw new IllegalArgumentException(
						String.format("Cannot remove column named \"%s\" because there is no such column.", 
								attr));
		}
	}
	
	void deleteColumns(List<String> ignoredAttributes, Matrix dataset)
	{
		if (ignoredAttributes == null || ignoredAttributes.size() == 0)
			return;
		if (ignoredAttributes.size() == 0)
			throw new IllegalArgumentException("No indexes given for removal.");
		if (Helper.isInteger(ignoredAttributes.get(0)))
		{
			int[] indexes = new int[ignoredAttributes.size()];
			for (int i = 0; i < indexes.length; i++)
			{
				indexes[i] = Integer.parseInt(ignoredAttributes.get(i));
			}
			deleteColumnsByIndex(indexes, dataset);
		}
		else
		{
			deleteColumnsByName(ignoredAttributes, dataset);
		}
	}
	
	/**
	 * Reorders the columns of labels. 
	 * @return If the columns a reordered, a new Matrix is returned. Otherwise labels is returned.
	 */
	static Matrix reorderLabelColumns(Matrix labels, List<Integer> labelOrder)
	{
		if (labelOrder == null)
			return labels;
		if (labelOrder.size() != labels.cols())
			throw new IllegalArgumentException("Must specify as many label order indexes as label columns.");
		if (labelOrder.size() == 0)
			throw new IllegalArgumentException("Must specify at least one index in label order.");
		// Verify labelOrder.
		{
			List<Integer> copy = new ArrayList<Integer>(labelOrder);
			Collections.sort(copy);
			for (int i = 0; i < copy.size(); i++)
				if (copy.get(i) != i)
					throw new IllegalArgumentException(String.format("Column reordering is missing index %s.", i));
		}
		Matrix result = new Matrix();
		for (int i = 0; i < labels.cols(); i++)
		{
			result.copyColumns(labels, labelOrder.get(i), 1);
		}

		return result;
	}

	
	public static Matrix fillOrRemoveUnknownData(List<String> unknownFillerNames, Matrix dataset)
	{
		for (String name : unknownFillerNames)
		{
			if (name.equals("none"))
			{
			}
			else if (name.equals("remove_labels"))
			{
				dataset = UnknownRemover.removeRowsWithUnknownOutputs(dataset);
			}
			else if (name.equals("mean_mode"))
			{
				MeanModeUnknownFiller filler = new MeanModeUnknownFiller();
				Pair<Matrix> inputsAndLabels = dataset.splitInputsAndLabels();
				filler.initialize(inputsAndLabels.getFirst(), inputsAndLabels.getSecond());
				Matrix datasetFiltered= filler.filterAllInputs(inputsAndLabels.getFirst());
				datasetFiltered.copyColumns(inputsAndLabels.getSecond(), 0, inputsAndLabels.getSecond().cols());
				datasetFiltered.setNumLabelColumns(dataset.getNumLabelColumns());
				dataset = datasetFiltered;
			}
			else
			{
				throw new IllegalArgumentException(String.format("Unrecognized unknown filler name: %s", 
						name));
			}
		}
		
		return dataset;
	}
	
	public static void testPrivateMethods() throws FileNotFoundException
	{
		// Test oversample.
		Matrix data = new Matrix();
		data.loadFromArffFile("Datasets/test/oversample.arff");
		data = data.oversample(new Random());
		Counter<Double> counter = new Counter<Double>();
		for (int r : new Range(data.rows()))
		{
			counter.increment(data.row(r).get(1));
			if (data.row(r).get(1) == 1.0)
				assertTrue(data.row(r).get(0) == 0);
		}
		assertEquals(3, counter.getCount(1.0));
		assertEquals(3, counter.getCount(0.0));
		assertEquals(6, data.rows());
	}

	public static void main(String[] args) throws Exception
	{
		Logger.addLoggingClassName(MLSystemsManager.class);
		MLSystemsManager ml = new MLSystemsManager();
		ml.run(args, null);
		Plotter.generateAllPlots();
	}
}
