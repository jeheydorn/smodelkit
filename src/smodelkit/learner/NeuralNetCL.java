package smodelkit.learner;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.bridj.Pointer;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.util.IOUtils;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.evaluator.Evaluator;
import smodelkit.evaluator.MSE;
import smodelkit.evaluator.RelativeEntropy;
import smodelkit.learner.neuralnet.*;
import smodelkit.util.BoundsFloat;
import smodelkit.util.Helper;
import smodelkit.util.Logger;
import smodelkit.util.Plotter;
import smodelkit.util.Range;


/**
 * A multi-layer perceptron that uses error backpropegation. It uses online weight updates.
 * @author joseph
 *
 */
public class NeuralNetCL extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	final boolean PRINT_EPOCH_TIMES = true;
	final int EPOCH_PRINT_FREQUENCY = 1;
	final boolean SAVE_ERROR_RATES = false;
	// This is the weights of each layer of the network; the hidden and output layers. The last layer is the output layer.
	float[] layers;
	// The number of weights (including bias weights) in nodes in each layer.
	int[] nodeWeightCountsPerLayer;
	// The number of nodes per layer.
	int[] nodeCountsPerLayer;
	
	float improvementThreshold;
	float validationSetPercent;
	int maxEpochs;
	int maxEpochsWithoutImprovement;
	protected int[] hiddenLayerSizes;
	float [] hiddenLayerMultiples;
	int maxHiddenLayerSize;
	boolean includLabelsInHiddenLayerMultiples;
	protected float learningRate;
	/** Used when the learner is training. It works in the domain of the filtered
	 * inputs and labels.
	 */
	protected Evaluator trainEvaluator;
	boolean increaseContrastOfHiddenLayerOutputs;
	Integer epochSize;
	Integer minEpochSize;
	private boolean normalizePredictions;
	private boolean softmax;
	private String hiddenLayerNodeType;
	private Pointer<Float> networkWeightsPtr;
	private CLBuffer<Float> networkWeightsBuf;
	private CLBuffer<Integer> nodeWeightCountsPerLayerBuf;
	private CLBuffer<Integer> nodeCountsPerLayerBuf;
	private CLBuffer<Float> outputsBuf;
	private CLBuffer<Float> errorsBuf;
	private CLQueue queue;
	private CLContext context;
	private CLKernel calcOutputsForLayer;
	
	
	public NeuralNetCL()
	{
		
	}
	
	@Override
	public void configure(JSONObject settings)
	{
		int[] hiddenLayerSizes = null;
		checkNullableArgumentIsPresent(settings, "hiddenLayerSizes");
		if (settings.get("hiddenLayerSizes") != null)
				hiddenLayerSizes = Helper.JSONArrayToIntArray((JSONArray)settings.get("hiddenLayerSizes"));
		float[] hiddenLayerMultiples = null;
		checkNullableArgumentIsPresent(settings, "hiddenLayerMultiples");
		if (settings.get("hiddenLayerMultiples") != null)
			hiddenLayerMultiples = Vector.convertToFloats(Helper.JSONArrayToDoubleArray((JSONArray)settings.get("hiddenLayerMultiples")));
		Long maxHiddenLayerSizeLong = (Long)settings.get("maxHiddenLayerSize");
		Integer maxHiddenLayerSize = maxHiddenLayerSizeLong != null ? maxHiddenLayerSizeLong.intValue() : null;
		float validationSetPercent = (Float)(float)(double)settings.get("validationSetPercent");
		float learningRate = (Float)(float)(double)settings.get("learningRate");
		float improvementThreshold = (Float)(float)(double)settings.get("improvementThreshold");
		
		if (settings.containsKey("momentum") && (Double)settings.get("momentum") != 0.0)
			throw new IllegalArgumentException("Momentum is not implemented in " + this.getClass().getSimpleName() + ".");
		
		checkNullableArgumentIsPresent(settings, "maxEpochs");
		int maxEpochs = Integer.MAX_VALUE;
		if (settings.get("maxEpochs") != null)
		{
			maxEpochs = (int)(long)settings.get("maxEpochs");
		}
		
		checkNullableArgumentIsPresent(settings, "hiddenLayerSizes");
		int maxEpochsWithoutImprovement = Integer.MAX_VALUE;
		if (settings.get("maxEpochsWithoutImprovement") != null)
			maxEpochsWithoutImprovement = (int)(long)settings.get("maxEpochsWithoutImprovement");
		
		boolean includLabelsInHiddenLayerMultiples 
				= (boolean)(Boolean)settings.get("includeLabelsInHiddenLayerMultiples");
		
		if ((hiddenLayerSizes != null) == (hiddenLayerMultiples != null))
			throw new IllegalArgumentException("NeuralNet needs exactly one of hiddenLayerSizes or " +
					"hiddenLayerMultiples. The other must be set to null.");
				
		boolean increaseContrastOfHiddenLayerOutputs = (boolean)(Boolean)settings.get("increaseContrastOfHiddenLayerOutputs");
		Long epochSizeLong = (Long)settings.get("epochSize");
		Integer epochSize = epochSizeLong != null ? epochSizeLong.intValue() : null;
		Long minEpochSizeLong = (Long)settings.get("minEpochSize");
		Integer minEpochSize = minEpochSizeLong != null ? minEpochSizeLong.intValue() : null;
		boolean normalizePredictions = (boolean)(Boolean)settings.get("normalizePredictions");
		String outputLayerNodeType = (String)settings.get("outputLayerNodeType");
		String hiddenLayerNodeType = (String)settings.get("hiddenLayerNodeType");
		
		configure(learningRate, hiddenLayerSizes, hiddenLayerMultiples, maxHiddenLayerSize, 
				validationSetPercent,
				improvementThreshold, maxEpochs, maxEpochsWithoutImprovement, includLabelsInHiddenLayerMultiples,
				increaseContrastOfHiddenLayerOutputs, epochSize, minEpochSize, 
				normalizePredictions, outputLayerNodeType, hiddenLayerNodeType);
	}
		
	/**
	 * 
	 * @param filter
	 * @param r
	 * @param learningRate
	 * @param hiddenLayerSizes Gives the size of each hidden layer as a number of nodes. The number of
	 * hidden layers will be hiddenLayersSizes.length.
	 * @param hiddenLayerMultiples  Gives the size of each hidden layer as a multiple of the number
	 * of features. Partial numbers of nodes will be rounded to integers. When NominalToCatagorical
	 * is used, the number of features used is the number before that filter was applied. One of 
	 * hiddenLayerSizes and hiddenLayerMultiples must be null.
	 * @param maxHiddenLayerSize Values for numbers of nodes in hiddenLayerSizes will be set to at most this.
	 * @param momentum
	 * @param validationSetPercent
	 * @param improvementThreshold
	 * @param maxEpochs
	 * @param maxEpochsWithoutImprovement
	 * @param epochSize  If not null, it will be used instead of the dataset size in the stopping criteria.
	 * @param minEpochSize If not null, if the dataset size is less than this value, then epochSize will be 
	 * set to this value. minEpochSize and epochSize cannot both be non-null.
	 * @param increaseContrastOfHiddenLayerOutputs If true then the inputs to hidden layer nodes will have increased
	 * contrast. This helps to speed up learner with 3 or more hidden layers.
	 * @param epochSize The size of an epoch. If null, this will be the training set size.
	 * @param normalizePredictions If true, then the weights assigned
	 * to each output in innerGetOutputWeights will be normalized to sum to 1.
	 * @param outputLayerNodeType The type of node in the output layer. Values are "sigmoidWithMSE" 
	 * and "softmaxWithRelativeEntropy". This option also determines the type of error measured on the validation set
	 * for the stopping criteria.
	 * @param hiddenLayerNodeType The type of node to use in the hidden layers. Values are "sigmoid" and "softsign".
	 */
	public void configure(float learningRate, int[] hiddenLayerSizes, 
			float[] hiddenLayerMultiples,
			Integer maxHiddenLayerSize, 
			float validationSetPercent, float improvementThreshold, int maxEpochs, 
			int maxEpochsWithoutImprovement, boolean includLabelsInHiddenLayerMultiples,
			boolean increaseContrastOfHiddenLayerOutputs, Integer epochSize, Integer minEpochSize,
			boolean normalizePredictions, String outputLayerNodeType, String hiddenLayerNodeType)
	{
		this.learningRate = learningRate;
		this.validationSetPercent = validationSetPercent;
		this.improvementThreshold = improvementThreshold;
		this.maxEpochs = maxEpochs;
		this.maxEpochsWithoutImprovement = maxEpochsWithoutImprovement;
		this.hiddenLayerSizes = hiddenLayerSizes;
		this.hiddenLayerMultiples = hiddenLayerMultiples;
		this.maxHiddenLayerSize = maxHiddenLayerSize == null ? Integer.MAX_VALUE : this.maxHiddenLayerSize;
		this.includLabelsInHiddenLayerMultiples = includLabelsInHiddenLayerMultiples;
		this.increaseContrastOfHiddenLayerOutputs = increaseContrastOfHiddenLayerOutputs;
		this.epochSize = epochSize;
		this.minEpochSize = minEpochSize;
		this.normalizePredictions = normalizePredictions;
		
		if (outputLayerNodeType.equals("sigmoidWithMSE"))
		{
			this.softmax = false;
		}
		else if (outputLayerNodeType.equals("softmaxWithRelativeEntropy"))
		{
			this.softmax = true;
		}
		else
		{
			throw new IllegalArgumentException("Unknown output layer node type: " + outputLayerNodeType);
		}
		
		this.hiddenLayerNodeType = hiddenLayerNodeType;
				
		setupTrainingEvaluator();
		varifyArgs();
	}
	
	protected void setupTrainingEvaluator()
	{
		if (softmax)
			trainEvaluator = new RelativeEntropy();
		else
			this.trainEvaluator = new MSE();
	}

	private void varifyArgs()
	{		
		if (learningRate == 0)
			throw new IllegalArgumentException();
		if (maxEpochs < 1)
			throw new IllegalArgumentException("maxEpochs must be at least 1.");
		if (maxEpochsWithoutImprovement < 1)
			throw new IllegalArgumentException("maxEpochsWithoutImprovement must be at least 1.");	
		if (epochSize != null && epochSize < 1)
			throw new IllegalArgumentException("epochSize must be at least 1 if given.");
		if (epochSize != null && minEpochSize != null)
			throw new IllegalArgumentException("epochSize and minEpochSize cannot both be non-null.");
		if (minEpochSize != null && minEpochSize < 1)
			throw new IllegalArgumentException("minEpochSize must be at least 1 if given.");
		
		if (!hiddenLayerNodeType.equals("sigmoid") && !hiddenLayerNodeType.equals("softsign"))
			throw new IllegalArgumentException("Unrecognized hidden layer node type: " + hiddenLayerNodeType);
	}
	
	public void innerTrain(Matrix inputs, Matrix labels)
	{		
		if (labels.isContinuous(0) && labels.getNumCatagoricalCols().size() == 0)
		{
			throw new UnsupportedOperationException("To support a numeric target, I need to implement a linear node.");
		}
		
		Logger.indent();
		Logger.println("NeuralNet verbose output: ");
		Logger.println("max epochs: " + maxEpochs);
		Logger.println("max epochs without improvement: " + maxEpochsWithoutImprovement);
		Logger.println("improvementThreshold: " + improvementThreshold);
		Logger.println("epochSize: " + epochSize);
		Logger.println("learning rate: " + learningRate);
		Logger.println("validation set %: " + validationSetPercent);
		Logger.println("increasContrastOfHiddenLayerInputs: " + increaseContrastOfHiddenLayerOutputs);
		
		// I'm copying these so that I don't shuffle the original inputs and labels.
		Matrix inputsTemp = new Matrix(inputs);
		Matrix labelsTemp = new Matrix(labels);
		inputs = null;
		labels = null;
		inputsTemp.shuffle(rand, labelsTemp);
	
		// Create validation set
		Matrix[] sets = createValidationSet(inputsTemp, labelsTemp, validationSetPercent);
		Matrix tInputs = sets[0];
		Matrix tLabels = sets[1];
		Matrix vInputs = sets[2];
		Matrix vLabels = sets[3];
		
		if (vInputs.rows() == 0 && validationSetPercent > 0)
		{
			// This means that the dataset so small that the validation set is empty. Validate on the training data.
			vInputs = tInputs;
			vLabels = tLabels;
		}
		else if(vInputs.rows() == 0 && maxEpochsWithoutImprovement != Integer.MAX_VALUE)
		{
			throw new IllegalArgumentException("If maxEpochsWithoutImprovement is not null, "
					+ "validationSetPercent cannot be 0.");
		}
		
		inputsTemp = null;
		labelsTemp = null;

		if (epochSize == null)
		{
			if (minEpochSize == null)
				epochSize = tInputs.rows();
			else
				epochSize = Math.max(minEpochSize, tInputs.rows());
		}

		if (hiddenLayerSizes != null)
			createNetwork(tInputs, tLabels.cols(), hiddenLayerSizes);
		else
			createNetwork(tInputs, tLabels, tLabels.cols());

		initializeWeights();

		if (layers != null)
		{
			Logger.println("Network input count: " + nodeWeightCountsPerLayer[0]); // -1 for bias weight.
			Logger.print("Layer sizes (the output layer is last): ");
			for (int i = 0; i < nodeCountsPerLayer.length; i++)
				Logger.print(nodeCountsPerLayer[i] + " ");
			Logger.println();
		}
		
		if (maxEpochs == 0 || maxEpochsWithoutImprovement == 0)
			return;

//		Logger.println("Weight before training: ");
//		printWeights();

		// A copy of the network layers from the time they did best on a validation set.
		float[] savedLayers = (float[]) Helper.deepCopy(layers);
		int epochOfSavedWeights = 0;

		float evaluation = 0;
		float lastEvaluation = 0;
		if (tLabels.isContinuous(0))
		{
			lastEvaluation = Float.MAX_VALUE;
		}
		int count = 0;
		int totalCount = 0;
		int nextInstanceIndex = 0;
		long timeBefore = System.currentTimeMillis();
		do
		{
			count++;
			totalCount++;

			context = JavaCL.createBestContext();

	        String src;
			try
			{
				src = IOUtils.readText(Paths.get("neuralnet.cl").toFile());
			} catch (IOException e)
			{
				throw new RuntimeException(e);
			}
	        CLProgram program = context.createProgram(src);

	        calcOutputsForLayer = program.createKernel("calcOutputsForLayer");
            CLKernel calcOutputLayerErrors = program.createKernel("calcOutputLayerErrors");
	        CLKernel calcHiddenLayerErrors = program.createKernel("calcHiddenLayerErrors");
	        CLKernel updateWeights = program.createKernel("updateWeights");

		    queue = context.createDefaultQueue();
		    
		    // Create buffers.
	        ByteOrder byteOrder = context.getByteOrder();

	        networkWeightsPtr = Pointer.pointerToFloats(layers).order(byteOrder);
	        networkWeightsBuf = context.createBuffer(Usage.InputOutput, networkWeightsPtr);
	        
	        nodeWeightCountsPerLayerBuf = context.createIntBuffer(Usage.Input, 
	        		Pointer.pointerToInts(nodeWeightCountsPerLayer).order(byteOrder));
	        
	        nodeCountsPerLayerBuf = context.createBuffer(Usage.Input, 
	        		Pointer.pointerToInts(nodeCountsPerLayer).order(byteOrder));
	        
	        outputsBuf = context.createBuffer(Usage.Output, Float.class, 
	        		Arrays.stream(nodeCountsPerLayer).sum());

	        errorsBuf = context.createBuffer(Usage.Output, Float.class, 
	        		Arrays.stream(nodeCountsPerLayer).sum());

			doEpoch(tInputs, tLabels, nextInstanceIndex, calcOutputsForLayer, calcOutputLayerErrors, 
					calcHiddenLayerErrors, updateWeights, context, queue, networkWeightsBuf, 
					nodeWeightCountsPerLayerBuf, nodeCountsPerLayerBuf, outputsBuf, errorsBuf);
			
			nextInstanceIndex = (nextInstanceIndex + epochSize) % tInputs.rows();
			
			if (PRINT_EPOCH_TIMES)
			{
				long timeAfter = System.currentTimeMillis();
				Logger.println("Epoch " + (totalCount) + " time: " + (timeAfter - timeBefore)/1000.0 + " seconds");
				timeBefore = timeAfter;
			}

			if (vInputs.rows() > 0)
			{
				evaluation = (float)(double)Evaluator.runEvaluators(vInputs, vLabels, this, false, 
						Collections.singletonList(trainEvaluator))
						.getScores(trainEvaluator.getClass()).get(0);
				
				if (SAVE_ERROR_RATES)
				{
					Plotter.addDatumForLinePlot(trainEvaluator.getClass().getSimpleName(),
							evaluation, "Epoch", trainEvaluator.getClass().getSimpleName());
					
				}
							
				float improvement = trainEvaluator.higherScoresAreBetter() ? 
						evaluation - lastEvaluation : lastEvaluation - evaluation;
				
				if(improvement > improvementThreshold)
				{
					count = 0;
					lastEvaluation = evaluation;
					Logger.println(String.format("Error improved to: %.5f on epoch: %s", evaluation, totalCount));
					
					copyWeights(savedLayers, layers);
					epochOfSavedWeights = totalCount;
				}
			}
		}
		while((evaluation < 1 || tLabels.isContinuous(0)) && count < maxEpochsWithoutImprovement && totalCount < maxEpochs);
		
		Logger.println();
	
		// Tell why we stopped training.
		if (!tLabels.isContinuous(0) && evaluation == 1)
		{
			Logger.println("Stopping training because predictive accuracy is 100%");
		}
		else if (count == maxEpochsWithoutImprovement)
		{
			Logger.println("Stopping training because max epochs without improvement was reached.");
		}
		else if (totalCount == maxEpochs)
		{
			Logger.println("Stopping training because max epochs was reached");
			Logger.println("Max epochs = " + maxEpochs);
		}
		else
		{
			throw new IllegalStateException("I don't know why I stopped.");
		}
		Logger.println("Epochs run: " + totalCount);
		
//		Logger.println("Weight after training: ");
//		printWeights();
		
		if (vInputs.rows() > 0 && savedLayers != null)
		{
			Logger.println("Restoring weights to those from epoch " + epochOfSavedWeights);
			copyWeights(layers, savedLayers);
			
			// Copy the weights to the GPU. TODO
			//CLEvent.waitFor(networkWeightsBuf.write(queue, networkWeightsPtr, false));
		}
		
		Logger.unindent();
	}

	protected void doEpoch(Matrix inputs, Matrix labels, int nextIndex, CLKernel calcOutputsForLayer, 
			CLKernel calcOutputLayerErrors,  CLKernel calcHiddenLayerErrors, CLKernel updateWeights,
			CLContext context, CLQueue queue,
			CLBuffer<Float> networkWeightsBuf, CLBuffer<Integer> nodeWeightCountsPerLayerBuf,
			CLBuffer<Integer> nodeCountsPerLayerBuf, CLBuffer<Float> outputsBuf, CLBuffer<Float> errorsBuf)
	{	 		
        ByteOrder byteOrder = context.getByteOrder();

		CLEvent event = null;
        for (int instanceRow : new Range(nextIndex, nextIndex + epochSize))
		{
	        CLBuffer<Float> inputsBuf  = context.createFloatBuffer(Usage.Input, 
	        		Pointer.pointerToFloats(inputs.row(instanceRow).getValuesFloat()).order(byteOrder));

	        // Calculate outputs for each node.
  			for (int layerIndex : new Range(nodeCountsPerLayer.length))
   			{	
 		        calcOutputsForLayer.setArgs(inputsBuf, inputs.cols(),
		        		layerIndex, networkWeightsBuf, nodeWeightCountsPerLayerBuf, 
		        		nodeCountsPerLayerBuf, outputsBuf);
  		        if (event == null)
  		        	event = calcOutputsForLayer.enqueueNDRange(queue, new int[] {nodeCountsPerLayer[layerIndex]});
  		        else
  		        	event = calcOutputsForLayer.enqueueNDRange(queue, new int[] {nodeCountsPerLayer[layerIndex]}, event);  
   			}
  			
  			// Calculate errors for the output layer nodes.
	        CLBuffer<Float> targetsBuf  = context.createFloatBuffer(Usage.Input, 
	        		Pointer.pointerToFloats(labels.row(instanceRow).getValuesFloat()).order(byteOrder));
 			calcOutputLayerErrors.setArgs(nodeCountsPerLayer.length - 1, networkWeightsBuf, nodeWeightCountsPerLayerBuf,
  					nodeCountsPerLayerBuf, outputsBuf, targetsBuf, errorsBuf);
  			event = calcOutputLayerErrors.enqueueNDRange(queue, 
  					new int[] {nodeCountsPerLayer[nodeCountsPerLayer.length - 1]}, event);
  			
  			// Calculate errors for hidden layer nodes.
  			for (int layerIndex = nodeCountsPerLayer.length - 2; layerIndex >= 0; layerIndex--)
   			{				
  				calcHiddenLayerErrors.setArgs(layerIndex, networkWeightsBuf, nodeWeightCountsPerLayerBuf,
  	  					nodeCountsPerLayerBuf, outputsBuf, errorsBuf);
 	        	event = calcHiddenLayerErrors.enqueueNDRange(queue, new int[] {nodeCountsPerLayer[layerIndex]}, event); 		        
   			}
  			
  			// Update weights.
  			for (int layerIndex : new Range(nodeCountsPerLayer.length))
   			{
 		        updateWeights.setArgs(layerIndex, inputsBuf, inputs.cols(), networkWeightsBuf, 
 		        		nodeWeightCountsPerLayerBuf,
  						nodeCountsPerLayerBuf, learningRate, errorsBuf, outputsBuf);
 	        	event = updateWeights.enqueueNDRange(queue, new int[] {nodeCountsPerLayer[layerIndex]}, event); 		          				
   			}
		}
        
        Pointer<Float> readPtr = networkWeightsBuf.read(queue, event);
        for (int j = 0; j < layers.length; j++)
        {
        	layers[j] = readPtr.get(j);
        }
	}
	
	private void initializeWeights()
	{
		final float standardDeviation = 0.1f;
		for (int i : new Range(layers.length))
		{
			layers[i] = (float)rand.nextGaussian() * standardDeviation;
		}
	}
	
	private void setWeight(int layerIndex, int nodeNumber, int weightNumber, float value)
	{
		int layerStart = new Range(layerIndex).stream().mapToInt(prevLayer -> nodeCountsPerLayer[prevLayer]).sum();
		
		layers[layerStart + nodeNumber * nodeWeightCountsPerLayer[layerIndex] + weightNumber] = value;
	}

	/**
	 * Copies all weight values from source to dest.
	 */
	private void copyWeights(float[] dest, float[] source)
	{
		assert dest.length == source.length;
		for (int i = 0; i < source.length; i++)
			{
				dest[i] = source[i];
			}
	}


	private float[][] extractOutputs(Pointer<Float> outPtr)
	{
        float[][] outputs = generateNetworkSizeArray();
        int ptrIndex = 0;
        for (int i = 0; i < nodeCountsPerLayer.length; i++)
	        for (int j = 0; j < nodeCountsPerLayer[i]; j++)
	        {
	        	outputs[i][j] = outPtr.get(ptrIndex);
	        	ptrIndex++;
	        }
	    return outputs;
	}
	
	void printWeights()
	{
		int layerStart = 0;
		for(int i = 0; i < layers.length; i++)
		{
			float[] curLayer = Arrays.copyOfRange(layers, layerStart, layerStart + nodeCountsPerLayer[i]);
			Logger.println("Weights for layer " + i + ": ");
			for(int j = 0; j < curLayer.length; j++)
			{
				int start = j * nodeWeightCountsPerLayer[i];
				int end = start + nodeWeightCountsPerLayer[i];
				float[] nodeWeights = Arrays.copyOfRange(curLayer, start, end);
				Logger.println(Helper.printArray("node " + j, nodeWeights));
			}
			layerStart += nodeCountsPerLayer[i];
		}
		Logger.println();

	}
	
	protected float[][] generateNetworkSizeArray()
	{
		float[][] result = new float[nodeCountsPerLayer.length][];
		for (int i = 0; i < nodeCountsPerLayer.length; i++)
		{
			result[i] = new float[nodeCountsPerLayer[i]];
		}
		return result;
	}

	protected float[][] calcOutputs(Vector input)
	{
        CLBuffer<Float> inputsBuf = context.createFloatBuffer(Usage.Input, 
        		Pointer.pointerToFloats(input.getValuesFloat()).order(context.getByteOrder()));

        // Calculate outputs for each node.
        CLEvent event = null;
		for (int layerIndex : new Range(nodeCountsPerLayer.length))
		{
	        calcOutputsForLayer.setArgs(inputsBuf, input.size(),
        		layerIndex, networkWeightsBuf, nodeWeightCountsPerLayerBuf, 
        		nodeCountsPerLayerBuf, outputsBuf);
	        if (event == null)
	        	event = calcOutputsForLayer.enqueueNDRange(queue, new int[] {nodeCountsPerLayer[layerIndex]});
	        else
	        	event = calcOutputsForLayer.enqueueNDRange(queue, new int[] {nodeCountsPerLayer[layerIndex]}, event);
		}
		CLEvent.waitFor(event);
		
		Pointer<Float> outPtr = outputsBuf.read(queue, event);
        float[][] outputs = extractOutputs(outPtr);
        outPtr.release();
		
		return outputs;
	}
	
	private static void increaseContrast(float[] values, BoundsFloat bounds)
	{
		float minVal = Helper.min(values);
		float maxVal = Helper.max(values);
		float range = maxVal - minVal;
		float change = Math.min(minVal - bounds.lower, bounds.upper - maxVal);
		float scale = (2f*change + range)/ range;
		if (Float.isInfinite(scale))
			return;
		float middle = (maxVal + minVal) / 2f;//Helper.mean(outputs[i]);
							
		for(int j = 0; j < values.length; j++)
		{
			values[j] = (values[j] - middle)*scale + middle;
		}	
	}

	public Vector innerPredict(Vector input)
	{
		float[][] outputs = calcOutputs(input);

		return Vector.create(outputs[outputs.length - 1]);
	}
	
	@Override
	public List<double[]> innerPredictOutputWeights(Vector input)
	{
		float[][] outputs = calcOutputs(input);
		float[] weights = outputs[outputs.length - 1];


		if (weights.length == 1)
		{
			// This means that the target is a binary output. For the result to be in the correct
			// format, I need to return 2 weights, one for each nominal value.
			
			float w = weights[0];
			// The squashing function should bound w between 0 and 1.
			assert w >= 0.0;
			assert w <= 1.0;

			float[] transformed = new float[2];
			transformed[0] = 1f - w;
			transformed[1] = w;
			weights = transformed;
		}
		else if (normalizePredictions)
		{
			if (!softmax)
			{
				// Increase the lower bound of the weights to be 0. I need to do this because I cannot
				// normalize an array with negative numbers.
				// TODO
//				for (int i = 0; i < weights.length; i++)
//					weights[i] -= kernelCreator.getNodeOutputRange().lower;
			}
			Helper.normalize(weights);
		}
		
		// For now I am returning only one float[]. This works fine fore single-dimensional classification,
		// but if I want the use this function with mutli-dimensional classification, I will need to chop
		// up this float[] into the pieces that correspond to each dimension. 
		
		return Collections.singletonList(Vector.convertToDoubles(weights));
	}
	
	public float dotProductErrorFromHigherLayer(int weightIndex, NeuralNode[] higherLayer,
			float[] higherLayerErrors)
	{
		float sum = 0;
		for (int i = 0; i < higherLayerErrors.length; i++)
		{
			sum += higherLayer[i].getWeight(weightIndex) * higherLayerErrors[i];
		}
		return sum;
	}
	
	/**
	 * Fills "layers" with nodes.  
	 * @param hiddenLayerSizes  Gives the size of each hidden layer in terms of number of nodes.
	 */
	void createNetwork(Matrix inputs, int numOutputs, int[] hiddenLayerSizes)
	{
		int layersSize = 0;
		nodeWeightCountsPerLayer = new int[hiddenLayerSizes.length + 1];
		nodeCountsPerLayer = new int[hiddenLayerSizes.length + 1];
		
		int prevLayerNodeCount;
		
		for(int i : new Range(hiddenLayerSizes.length))
		{
			if (hiddenLayerSizes[i] == 0)
				throw new IllegalArgumentException("A hidden layer cannot have 0 nodes.");
			
			// Each node has 1 input from every node in the layer closer
			// to the inputs, except those receiving the features as inputs.
			int numInputs = i == 0 ? inputs.cols() : nodeCountsPerLayer[i - 1];
			
			nodeCountsPerLayer[i] = Math.min(maxHiddenLayerSize, hiddenLayerSizes[i]);
			nodeWeightCountsPerLayer[i] = numInputs + 1;
			layersSize += nodeWeightCountsPerLayer[i] * nodeCountsPerLayer[i];
		}
		
		// The output layer has 1 node per output.
		int numOutputLayerInputs = hiddenLayerSizes.length > 0 ? nodeCountsPerLayer[hiddenLayerSizes.length - 1] : inputs.cols();
		nodeWeightCountsPerLayer[nodeWeightCountsPerLayer.length - 1] = numOutputLayerInputs + 1;
		layersSize += nodeWeightCountsPerLayer[nodeWeightCountsPerLayer.length - 1] * numOutputs;
		nodeCountsPerLayer[nodeCountsPerLayer.length - 1] = numOutputs;
		layers = new float[layersSize];
	}
	
	/**
	 * Fills "layers" with nodes.  
	 * hiddenLayerMultiples gives the size of each hidden layer as a multiple of the number of features or the number
	 * of features + labels.
	 * Partial numbers of nodes will be rounded to integers.
	 */
	void createNetwork(Matrix inputs, Matrix labels, int numOutputs)
	{		
		int[] hiddenLayerSizes = new int[hiddenLayerMultiples.length];
		for (int i = 0; i < hiddenLayerMultiples.length; i++)
		{			
			if (inputs.getFilteredNominalColsTotal() > 0)
			{
				int labelsAdded = includLabelsInHiddenLayerMultiples ? labels.getFilteredNominalColsTotal() : 0;
				hiddenLayerSizes[i] = (int)Math.round((inputs.getFilteredNominalColsTotal() + labelsAdded) 
						* hiddenLayerMultiples[i]);
			}
			else
			{
				// This is a separate case because inputs.getNominalColsTotal() returns zero when the
				// NominalToCategorical filter is not used.
				
				int labelsAdded = includLabelsInHiddenLayerMultiples ? labels.cols() : 0;
				hiddenLayerSizes[i] = (int)Math.round((inputs.cols() + labelsAdded)* hiddenLayerMultiples[i]);
			}
		}
		createNetwork(inputs, numOutputs, hiddenLayerSizes);
	}

	
	@Override
	protected boolean canImplicitlyHandleNominalFeatures()
	{
		return false;
	}
	
	@Override
	protected boolean canImplicitlyHandleContinuousFeatures()
	{
		return true;
	}
	
	@Override
	protected boolean canImplicitlyHandleNominalLabels()
	{
		// Technically NeuralNet could handle nominal labels, but doing so implies
		// an order in the nominal values, which doesn't make sense. 
		return false;
	}
	
	@Override
	protected boolean canImplicitlyHandleContinuousLabels()
	{
		return true;
	}
	
	@Override
	protected boolean canImplicitlyHandleUnknownInputs()
	{
		return false;
	}
	
	@Override
	protected boolean canImplicitlyHandleMultipleOutputs()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleUnknownOutputs()
	{
		return false;
	}
	
	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
	}
	
}
