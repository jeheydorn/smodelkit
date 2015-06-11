//package smodelkit.learner;
//import java.util.Collections;
//import java.util.List;
//
//import org.json.simple.JSONArray;
//import org.json.simple.JSONObject;
//
//import com.amd.aparapi.Kernel;
//
//import smodelkit.Matrix;
//import smodelkit.Vector;
//import smodelkit.evaluator.Evaluator;
//import smodelkit.evaluator.MSE;
//import smodelkit.evaluator.RelativeEntropy;
//import smodelkit.learner.neuralnet.*;
//import smodelkit.util.Bounds;
//import smodelkit.util.Helper;
//import smodelkit.util.Logger;
//import smodelkit.util.Plotter;
//import smodelkit.util.Range;
//
//
///**
// * A multi-layer perceptron that uses error backpropigation. It uses online weight updates.
// * @author joseph
// *
// */
//public class NeuralNetGPU extends SupervisedLearner
//{
//	private static final long serialVersionUID = 1L;
//	final boolean PRINT_EPOCH_TIMES = false;
//	final int EPOCH_PRINT_FREQUENCY = 1;
//	final boolean SAVE_ERROR_RATES = false;
//	// This is the weights of each layer of the network; the hidden and output layers. The last layer is the output layer.
//	double[][] layers;
//	// The number of weights (including bias weights) in nodes in each layer.
//	int[] nodeWeightCountsPerLayer;
//	// The number of nodes per layer.
//	int[] nodeCountPerLayer;
//	
//	double improvementThreshold;
//	double validationSetPercent;
//	int maxEpochs;
//	int maxEpochsWithoutImprovement;
//	protected int[] hiddenLayerSizes;
//	double [] hiddenLayerMultiples;
//	Integer maxHiddenLayerSize;
//	boolean includLabelsInHiddenLayerMultiples;
//	protected double learningRate;
//	/** Used when the learner is training. It works in the domain of the filtered
//	 * inputs and labels.
//	 */
//	protected Evaluator trainEvaluator;
//	boolean increaseContrastOfHiddenLayerOutputs;
//	Integer epochSize;
//	Integer minEpochSize;
//	private boolean normalizePredictions;
//	private boolean softmax;
//	private String hiddenLayerNodeType;
//	SigmoidNodeKernelCreator kernelCreator;
//	// TODO remove
//	int plotCount = 0;
//	long freq = Long.MAX_VALUE;
//	
//	
//	public NeuralNetGPU()
//	{
//		
//	}
//	
//	@Override
//	public void configure(JSONObject settings)
//	{
//		int[] hiddenLayerSizes = null;
//		checkNullableArgumentIsPresent(settings, "hiddenLayerSizes");
//		if (settings.get("hiddenLayerSizes") != null)
//				hiddenLayerSizes = Helper.JSONArrayToIntArray((JSONArray)settings.get("hiddenLayerSizes"));
//		double[] hiddenLayerMultiples = null;
//		checkNullableArgumentIsPresent(settings, "hiddenLayerMultiples");
//		if (settings.get("hiddenLayerMultiples") != null)
//			hiddenLayerMultiples = Helper.JSONArrayToDoubleArray((JSONArray)settings.get("hiddenLayerMultiples"));
//		Long maxHiddenLayerSizeLong = (Long)settings.get("maxHiddenLayerSize");
//		Integer maxHiddenLayerSize = maxHiddenLayerSizeLong != null ? maxHiddenLayerSizeLong.intValue() : null;
//		double validationSetPercent = (Double)settings.get("validationSetPercent");
//		double learningRate = (Double)settings.get("learningRate");
//		double improvementThreshold = (Double)settings.get("improvementThreshold");
//		
//		checkNullableArgumentIsPresent(settings, "maxEpochs");
//		int maxEpochs = Integer.MAX_VALUE;
//		if (settings.get("maxEpochs") != null)
//		{
//			maxEpochs = (int)(long)settings.get("maxEpochs");
//		}
//		
//		checkNullableArgumentIsPresent(settings, "hiddenLayerSizes");
//		int maxEpochsWithoutImprovement = Integer.MAX_VALUE;
//		if (settings.get("maxEpochsWithoutImprovement") != null)
//			maxEpochsWithoutImprovement = (int)(long)settings.get("maxEpochsWithoutImprovement");
//		
//		boolean includLabelsInHiddenLayerMultiples 
//				= (boolean)(Boolean)settings.get("includeLabelsInHiddenLayerMultiples");
//		
//		if ((hiddenLayerSizes != null) == (hiddenLayerMultiples != null))
//			throw new IllegalArgumentException("NeuralNet needs exactly one of hiddenLayerSizes or " +
//					"hiddenLayerMultiples. The other must be set to null.");
//				
//		boolean increaseContrastOfHiddenLayerOutputs = (boolean)(Boolean)settings.get("increaseContrastOfHiddenLayerOutputs");
//		Long epochSizeLong = (Long)settings.get("epochSize");
//		Integer epochSize = epochSizeLong != null ? epochSizeLong.intValue() : null;
//		Long minEpochSizeLong = (Long)settings.get("minEpochSize");
//		Integer minEpochSize = minEpochSizeLong != null ? minEpochSizeLong.intValue() : null;
//		boolean normalizePredictions = (boolean)(Boolean)settings.get("normalizePredictions");
//		String outputLayerNodeType = (String)settings.get("outputLayerNodeType");
//		String hiddenLayerNodeType = (String)settings.get("hiddenLayerNodeType");
//		
//		configure(learningRate, hiddenLayerSizes, hiddenLayerMultiples, maxHiddenLayerSize, 
//				validationSetPercent,
//				improvementThreshold, maxEpochs, maxEpochsWithoutImprovement, includLabelsInHiddenLayerMultiples,
//				increaseContrastOfHiddenLayerOutputs, epochSize, minEpochSize, 
//				normalizePredictions, outputLayerNodeType, hiddenLayerNodeType);
//	}
//		
//	/**
//	 * 
//	 * @param filter
//	 * @param r
//	 * @param learningRate
//	 * @param hiddenLayerSizes Gives the size of each hidden layer as a number of nodes. The number of
//	 * hidden layers will be hiddenLayersSizes.length.
//	 * @param hiddenLayerMultiples  Gives the size of each hidden layer as a multiple of the number
//	 * of features. Partial numbers of nodes will be rounded to integers. When NominalToCatagorical
//	 * is used, the number of features used is the number before that filter was applied. One of 
//	 * hiddenLayerSizes and hiddenLayerMultiples must be null.
//	 * @param maxHiddenLayerSize Values for numbers of nodes in hiddenLayerSizes will be set to at most this.
//	 * @param momentum
//	 * @param validationSetPercent
//	 * @param improvementThreshold
//	 * @param maxEpochs
//	 * @param maxEpochsWithoutImprovement
//	 * @param epochSize  If not null, it will be used instead of the dataset size in the stopping criteria.
//	 * @param minEpochSize If not null, if the dataset size is less than this value, then epochSize will be 
//	 * set to this value. minEpochSize and epochSize cannot both be non-null.
//	 * @param increaseContrastOfHiddenLayerOutputs If true then the inputs to hidden layer nodes will have increased
//	 * contrast. This helps to speed up learner with 3 or more hidden layers.
//	 * @param epochSize The size of an epoch. If null, this will be the training set size.
//	 * @param normalizePredictions If true, then the weights assigned
//	 * to each output in innerGetOutputWeights will be normalized to sum to 1.
//	 * @param outputLayerNodeType The type of node in the output layer. Values are "sigmoidWithMSE" 
//	 * and "softmaxWithRelativeEntropy". This option also determines the type of error measured on the validation set
//	 * for the stopping criteria.
//	 * @param hiddenLayerNodeType The type of node to use in the hidden layers. Values are "sigmoid" and "softsign".
//	 */
//	public void configure(double learningRate, int[] hiddenLayerSizes, 
//			double[] hiddenLayerMultiples,
//			Integer maxHiddenLayerSize, 
//			double validationSetPercent, double improvementThreshold, int maxEpochs, 
//			int maxEpochsWithoutImprovement, boolean includLabelsInHiddenLayerMultiples,
//			boolean increaseContrastOfHiddenLayerOutputs, Integer epochSize, Integer minEpochSize,
//			boolean normalizePredictions, String outputLayerNodeType, String hiddenLayerNodeType)
//	{
//		this.learningRate = learningRate;
//		this.validationSetPercent = validationSetPercent;
//		this.improvementThreshold = improvementThreshold;
//		this.maxEpochs = maxEpochs;
//		this.maxEpochsWithoutImprovement = maxEpochsWithoutImprovement;
//		this.hiddenLayerSizes = hiddenLayerSizes;
//		this.hiddenLayerMultiples = hiddenLayerMultiples;
//		this.maxHiddenLayerSize = maxHiddenLayerSize;
//		this.includLabelsInHiddenLayerMultiples = includLabelsInHiddenLayerMultiples;
//		this.increaseContrastOfHiddenLayerOutputs = increaseContrastOfHiddenLayerOutputs;
//		this.epochSize = epochSize;
//		this.minEpochSize = minEpochSize;
//		this.normalizePredictions = normalizePredictions;
//		
//		if (outputLayerNodeType.equals("sigmoidWithMSE"))
//		{
//			this.softmax = false;
//		}
//		else if (outputLayerNodeType.equals("softmaxWithRelativeEntropy"))
//		{
//			this.softmax = true;
//		}
//		else
//		{
//			throw new IllegalArgumentException("Unknown output layer node type: " + outputLayerNodeType);
//		}
//		
//		this.hiddenLayerNodeType = hiddenLayerNodeType;
//		
//		kernelCreator = new SigmoidNodeKernelCreator(); // TODO Make this a parameter.
//		
//		setupTrainingEvaluator();
//		varifyArgs();
//	}
//	
//	protected void setupTrainingEvaluator()
//	{
//		if (softmax)
//			trainEvaluator = new RelativeEntropy();
//		else
//			this.trainEvaluator = new MSE();
//	}
//
//	private void varifyArgs()
//	{		
//		if (learningRate == 0)
//			throw new IllegalArgumentException();
//		if (maxEpochs < 1)
//			throw new IllegalArgumentException("maxEpochs must be at least 1.");
//		if (maxEpochsWithoutImprovement < 1)
//			throw new IllegalArgumentException("maxEpochsWithoutImprovement must be at least 1.");	
//		if (epochSize != null && epochSize < 1)
//			throw new IllegalArgumentException("epochSize must be at least 1 if given.");
//		if (epochSize != null && minEpochSize != null)
//			throw new IllegalArgumentException("epochSize and minEpochSize cannot both be non-null.");
//		if (minEpochSize != null && minEpochSize < 1)
//			throw new IllegalArgumentException("minEpochSize must be at least 1 if given.");
//		
//		if (!hiddenLayerNodeType.equals("sigmoid") && !hiddenLayerNodeType.equals("softsign"))
//			throw new IllegalArgumentException("Unrecognized hidden layer node type: " + hiddenLayerNodeType);
//	}
//	
//	public void innerTrain(Matrix inputs, Matrix labels)
//	{		
//		if (labels.isContinuous(0) && labels.getNumCatagoricalCols().size() == 0)
//		{
//			// TODO Implement a linear unit for this case.
//			throw new UnsupportedOperationException("To support a numeric target, I need to implement a linear node.");
//		}
//		
//		Logger.indent();
//		Logger.println("NeuralNet verbose output: ");
//		Logger.println("max epochs: " + maxEpochs);
//		Logger.println("max epochs without improvement: " + maxEpochsWithoutImprovement);
//		Logger.println("improvementThreshold: " + improvementThreshold);
//		Logger.println("epochSize: " + epochSize);
//		Logger.println("learning rate: " + learningRate);
//		Logger.println("validation set %: " + validationSetPercent);
//		Logger.println("increasContrastOfHiddenLayerInputs: " + increaseContrastOfHiddenLayerOutputs);
//		
//		// I'm copying these so that I don't shuffle the original inputs and labels.
//		Matrix inputsTemp = new Matrix(inputs);
//		Matrix labelsTemp = new Matrix(labels);
//		inputs = null;
//		labels = null;
//		inputsTemp.shuffle(rand, labelsTemp);
//	
//		// Create validation set
//		Matrix[] sets = createValidationSet(inputsTemp, labelsTemp, validationSetPercent);
//		Matrix tInputs = sets[0];
//		Matrix tLabels = sets[1];
//		Matrix vInputs = sets[2];
//		Matrix vLabels = sets[3];
//		
//		if (vInputs.rows() == 0 && validationSetPercent > 0)
//		{
//			// This means that the dataset so small that the validation set is empty. Validate on the training data.
//			vInputs = tInputs;
//			vLabels = tLabels;
//		}
//		else if(vInputs.rows() == 0 && maxEpochsWithoutImprovement != Integer.MAX_VALUE)
//		{
//			throw new IllegalArgumentException("If maxEpochsWithoutImprovement is not null, "
//					+ "validationSetPercent cannot be 0.");
//		}
//		
//		inputsTemp = null;
//		labelsTemp = null;
//
//		if (epochSize == null)
//		{
//			if (minEpochSize == null)
//				epochSize = tInputs.rows();
//			else
//				epochSize = Math.max(minEpochSize, tInputs.rows());
//		}
//
//		if (hiddenLayerSizes != null)
//			createNetwork(tInputs, tLabels.cols(), hiddenLayerSizes);
//		else
//			createNetwork(tInputs, tLabels, tLabels.cols());
//
//
//		if (layers != null)
//		{
//			Logger.println("Network input count: " + nodeWeightCountsPerLayer[0]); // -1 for bias weight.
//			Logger.print("Layer sizes (the output layer is last): ");
//			for (int i = 0; i < layers.length; i++)
//				Logger.print(layers[i].length + " ");
//			Logger.println();
//		}
//		
//		if (maxEpochs == 0 || maxEpochsWithoutImprovement == 0)
//			return;
//
////		Logger.println("Weight before training: ");
////		printWeights();
//
//		// A copy of the network layers from the time they did best on a validation set.
//		double[][] savedLayers = (double[][]) Helper.deepCopy(layers);
//		int epochOfSavedWeights = 0;
//
//		double evaluation = 0;
//		double lastEvaluation = 0;
//		if (tLabels.isContinuous(0))
//		{
//			lastEvaluation = Double.MAX_VALUE;
//		}
//		int count = 0;
//		int totalCount = 0;
//		int nextInstanceIndex = 0;
//		double timeBefore = System.currentTimeMillis();
//		do
//		{
//			count++;
//			totalCount++;
//
////			if (totalCount % EPOCH_PRINT_FREQUENCY == 0)
////				Logger.println("Epoch number: " + totalCount);
//
//			doEpoch(tInputs, tLabels, nextInstanceIndex);
//			nextInstanceIndex = (nextInstanceIndex + epochSize) % tInputs.rows();
//			
//			if (PRINT_EPOCH_TIMES)
//			{
//				double timeAfter = System.currentTimeMillis();
//				Logger.println("Epoch time: " + (timeAfter - timeBefore)/1000.0 + " seconds");
//				timeBefore = timeAfter;
//			}
//
//			if (vInputs.rows() > 0)
//			{
//				evaluation = Evaluator.runEvaluators(vInputs, vLabels, this, false, 
//						Collections.singletonList(trainEvaluator))
//						.getScores(trainEvaluator.getClass()).get(0);
//				
//				if (SAVE_ERROR_RATES)
//				{
//					Plotter.addDatumForLinePlot(trainEvaluator.getClass().getSimpleName(),
//							evaluation, "Epoch", trainEvaluator.getClass().getSimpleName());
//					
//				}
//							
//				double improvement = trainEvaluator.higherScoresAreBetter() ? 
//						evaluation - lastEvaluation : lastEvaluation - evaluation;
//				
//				if(improvement > improvementThreshold)
//				{
//					count = 0;
//					lastEvaluation = evaluation;
//					Logger.println(String.format("Error improved to: %.5f on epoch: %s", evaluation, totalCount));
//	
//					copyWeights(savedLayers, layers);
//					epochOfSavedWeights = totalCount;
//				}
//			}
//		}
//		while((evaluation < 1 || tLabels.isContinuous(0)) && count < maxEpochsWithoutImprovement && totalCount < maxEpochs);
//		
//		Logger.println();
//	
//		// Tell why we stopped training.
//		if (!tLabels.isContinuous(0) && evaluation == 1)
//		{
//			Logger.println("Stopping training because predictive accuracy is 100%");
//		}
//		else if (count == maxEpochsWithoutImprovement)
//		{
//			Logger.println("Stopping training because max epochs without improvement was reached.");
//		}
//		else if (totalCount == maxEpochs)
//		{
//			Logger.println("Stopping training because max epochs was reached");
//			Logger.println("Max epochs = " + maxEpochs);
//		}
//		else
//		{
//			throw new IllegalStateException("I don't know why I stopped.");
//		}
//		Logger.println("Epochs run: " + totalCount);
//		
////		Logger.println("Weight after training: ");
////		printWeights();
//		
//		if (vInputs.rows() > 0 && savedLayers != null)
//		{
//			Logger.println("Restoring weights to those from epoch " + epochOfSavedWeights);
//			copyWeights(layers, savedLayers);
//		}
//		
//		
//		Logger.unindent();
//	}
//
//	/**
//	 * Copies all weight values from source to dest.
//	 */
//	private void copyWeights(double[][] dest, double[][] source)
//	{
//		for (int i = 0; i < source.length; i++)
//			for (int j = 0; j < source[i].length; j++)
//			{
//				dest[i][j] = source[i][j];
//			}
//	}
//
//	protected void doEpoch(Matrix inputs, Matrix labels, int nextIndex)
//	{	
//		for (int instanceRow : new Range(nextIndex, nextIndex + epochSize))
//		{
//			instanceRow %= inputs.rows();
//
//			// Calculate the output for every node
//			double[][] outputs = calcOutputs(inputs.row(instanceRow));
////			printVector("outputs: ", outputs[outputs.length - 1]);
//
//
//			// Calculate errors for every node
//			double[][] errors = generateNetworkSizeArray();
//			for(int i = layers.length -1; i >= 0 ; i--)
//			{
//				for(int j = 0; j < layers[i].length; j++)
//				{
//					if (i == layers.length - 1)
//					{
//						// output node
//						errors[i][j] = layers[i][j].calcOutputNodeError(labels.row(instanceRow).get(j), outputs[i][j]);
//					}
//					else
//					{
//						// hidden node
//						double errorFromHigherLayer = dotProductErrorFromHigherLayer(j, layers[i + 1], errors[i + 1]);
//						errors[i][j] = layers[i][j].calcHiddenNodeError(errorFromHigherLayer, outputs[i][j]);
//					}
//				}
//				if (plotCount % freq == 0)
//				{
//					Plotter.addDatumForLinePlot("layer_" + i + "_error", errors[i], "prediction", "error");
//				}
//			}
//
//			// Update all weights
//			for(int i = 0; i < layers.length; i++)
//			{
//				Vector input = i == 0 ? inputs.row(instanceRow) : new Vector(outputs[i - 1]);
//				double instanceWeight = inputs.row(instanceRow).getWeight();
//				for(int j = 0; j < layers[i].length; j++)
//				{
//					if (i == outputs.length - 1)
//					{
//						layers[i][j].updateWeights(input, errors[i][j], learningRate * instanceWeight);
//					}
//					else
//					{
//						layers[i][j].updateWeights(input, errors[i][j], learningRate * instanceWeight);
//					}
//				}
//			}
//		}
//	}
//	
//	protected double[][] generateNetworkSizeArray()
//	{
//		double[][] result = new double[layers.length][];
//		for (int i = 0; i < layers.length; i++)
//		{
//			result[i] = new double[layers[i].length];
//		}
//		return result;
//	}
//
//	protected double[][] calcOutputs(Vector input)
//	{
//		plotCount++;
//
//		double[][] outputs = generateNetworkSizeArray();
//		for(int i = 0; i < layers.length; i++)
//		{
//			Vector inputs = i == 0 ? input : new Vector(outputs[i - 1]);
//			int numWeights = nodeWeightCountsPerLayer[i];
//			
//			Kernel kernel = kernelCreator.createHiddenLayerOutputKernel(inputs.getValues(), layers[i], 
//			 		numWeights, outputs[i]);
//
//			kernel.execute(com.amd.aparapi.Range.create(outputs[i].length));
//			kernel.dispose();
//			
//			if (increaseContrastOfHiddenLayerOutputs)
//			{
//				// Don't increase the contrast of the output layer outputs.
//				if (i + 1 < layers.length)
//				{
//					Bounds nodeBounds = kernelCreator.getNodeOutputRange();
//					increaseContrast(outputs[i], nodeBounds);
//				}
//			}
//			if (plotCount % freq == 0)
//			{
//				Plotter.addDatumForLinePlot("layer_" + i, outputs[i], "prediction", "activation");
//				//Plotter.addDatumForLinePlot("layer_" + i + "_weights", layers[i][0].getWeights(), "prediction", "weight");
//			}
//			
//		}
//		if (softmax)
//			RelativeEntropy.softmaxInPlace(outputs[outputs.length - 1]);
//		return outputs;
//	}
//	
//	private static void increaseContrast(double[] values, Bounds bounds)
//	{
//		double minVal = Helper.min(values);
//		double maxVal = Helper.max(values);
//		double range = maxVal - minVal;
//		double change = Math.min(minVal - bounds.lower, bounds.upper - maxVal);
//		double scale = (2.0*change + range)/ range;
//		if (Double.isInfinite(scale))
//			return;
//		double middle = (maxVal + minVal) / 2.0;//Helper.mean(outputs[i]);
//							
//		for(int j = 0; j < values.length; j++)
//		{
//			values[j] = (values[j] - middle)*scale + middle;
//		}	
//	}
//
//	public Vector innerPredict(Vector input)
//	{
//		double[][] outputs = calcOutputs(input);
//
//		return new Vector(outputs[outputs.length - 1]);
//	}
//	
//	@Override
//	public List<double[]> innerPredictOutputWeights(Vector input)
//	{
//		double[][] outputs = calcOutputs(input);
//		double[] weights = outputs[outputs.length - 1];
//
//
//		if (weights.length == 1)
//		{
//			// This means that the target is a binary output. For the result to be in the correct
//			// format, I need to return 2 weights, one for each nominal value.
//			
//			double w = weights[0];
//			// The squashing function should bound w between 0 and 1.
//			assert w >= 0.0;
//			assert w <= 1.0;
//
//			double[] transformed = new double[2];
//			transformed[0] = 1.0 - w;
//			transformed[1] = w;
//			weights = transformed;
//		}
//		else if (normalizePredictions)
//		{
//			if (!softmax)
//			{
//				// Increase the lower bound of the weights to be 0. I need to do this because I cannot
//				// normalize an array with negative numbers.
//				for (int i = 0; i < weights.length; i++)
//					weights[i] -= layers[layers.length - 1][0].getOutputRange().lower;
//			}
//			Helper.normalize(weights);
//		}
//		
//		// For now I am returning only one double[]. This works fine fore single-dimensional classification,
//		// but if I want the use this function with mutli-dimensional classification, I will need to chop
//		// up this double[] into the pieces that correspond to each dimension. 
//		
//		return Collections.singletonList(weights);
//	}
//	
//	public double dotProductErrorFromHigherLayer(int weightIndex, Node[] higherLayer,
//			double[] higherLayerErrors)
//	{
//		double sum = 0;
//		for (int i = 0; i < higherLayerErrors.length; i++)
//		{
//			sum += higherLayer[i].getWeight(weightIndex) * higherLayerErrors[i];
//		}
//		return sum;
//	}
//
//	void printWeights()
//	{
//		for(int i = 0; i < layers.length; i++)
//		{
//
//			Logger.println("Weights for layer " + i + ": ");
//			for(int j = 0; j < layers[i].length; j++)
//			{
//				double[] nodeWeights = layers[i][j].getWeights();
//				Logger.println(Helper.printArray("node " + j, nodeWeights));
//			}
//		}
//		Logger.println();
//
//	}
//	
//	/**
//	 * Fills "layers" with nodes.  
//	 * @param hiddenLayerSizes  Gives the size of each hidden layer in terms of number of nodes.
//	 */
//	void createNetwork(Matrix inputs, int numOutputs, int[] hiddenLayerSizes)
//	{
//		layers = new Node[hiddenLayerSizes.length + 1][];
//		
//		for(int i = 0; i < layers.length - 1; i++)
//		{
//			if (hiddenLayerSizes[i] == 0)
//				throw new IllegalArgumentException("A hidden layer cannot have 0 nodes.");
//			layers[i] = new Node[maxHiddenLayerSize == null ?  hiddenLayerSizes[i] 
//					: Math.min(maxHiddenLayerSize, hiddenLayerSizes[i])];
//			
//			// Each node has 1 input from every node in the layer closer
//			// to the inputs, except those receiving the features as inputs.
//			int numInputs = i == 0 ? inputs.cols() : layers[i-1].length;
//
//			for(int j = 0; j < layers[i].length; j++)
//			{
//				if (hiddenLayerNodeType.equals("sigmoid"))
//					layers[i][j] = new SigmoidNode(rand, numInputs, momentum); // TODO Change back to SigmoidNode.
//				else if (hiddenLayerNodeType.equals("softsign"))
//					layers[i][j] = new SoftsignNode(rand, numInputs, momentum);
//				else
//					throw new IllegalArgumentException();
//			}
//		}
//		
//		// Create the output layer. It has 1 node per output.
//		layers[layers.length -1] = new Node[numOutputs];
//		for (int n = 0; n < layers[layers.length - 1].length; n++)
//		{
//			int numOutputLayerIntputs = layers.length > 1 ? layers[layers.length-2].length : inputs.cols();
//			if (softmax)
//				layers[layers.length - 1][n] = new SoftmaxNode(rand, numOutputLayerIntputs, momentum);
//			else
//				layers[layers.length - 1][n] = new SigmoidNode(rand, numOutputLayerIntputs, momentum); 
//		}
//	}
//	
//	/**
//	 * Fills "layers" with nodes.  
//	 * hiddenLayerMultiples gives the size of each hidden layer as a multiple of the number of features or the number
//	 * of features + labels.
//	 * Partial numbers of nodes will be rounded to integers.
//	 */
//	void createNetwork(Matrix inputs, Matrix labels, int numOutputs)
//	{		
//		int[] hiddenLayerSizes = new int[hiddenLayerMultiples.length];
//		for (int i = 0; i < hiddenLayerMultiples.length; i++)
//		{			
//			if (inputs.getFilteredNominalColsTotal() > 0)
//			{
//				int labelsAdded = includLabelsInHiddenLayerMultiples ? labels.getFilteredNominalColsTotal() : 0;
//				hiddenLayerSizes[i] = (int)Math.round((inputs.getFilteredNominalColsTotal() + labelsAdded) 
//						* hiddenLayerMultiples[i]);
//			}
//			else
//			{
//				// This is a separate case because inputs.getNominalColsTotal() returns zero when the
//				// NominalToCategorical filter is not used.
//				
//				int labelsAdded = includLabelsInHiddenLayerMultiples ? labels.cols() : 0;
//				hiddenLayerSizes[i] = (int)Math.round((inputs.cols() + labelsAdded)* hiddenLayerMultiples[i]);
//			}
//		}
//		createNetwork(inputs, numOutputs, hiddenLayerSizes);
//	}
//
//	
//	@Override
//	protected boolean canImplicitlyHandleNominalFeatures()
//	{
//		return false;
//	}
//	
//	@Override
//	protected boolean canImplicitlyHandleContinuousFeatures()
//	{
//		return true;
//	}
//	
//	@Override
//	protected boolean canImplicitlyHandleNominalLabels()
//	{
//		// Technically NeuralNet could handle nominal labels, but doing so implies
//		// an order in the nominal values, which doesn't make sense. 
//		return false;
//	}
//	
//	@Override
//	protected boolean canImplicitlyHandleContinuousLabels()
//	{
//		return true;
//	}
//	
//	@Override
//	protected boolean canImplicitlyHandleUnknownInputs()
//	{
//		return false;
//	}
//	
//	@Override
//	protected boolean canImplicitlyHandleMultipleOutputs()
//	{
//		return true;
//	}
//
//	@Override
//	protected boolean canImplicitlyHandleUnknownOutputs()
//	{
//		return false;
//	}
//	
//	@Override
//	public boolean canImplicitlyHandleInstanceWeights()
//	{
//		return true;
//	}
//	
//}
