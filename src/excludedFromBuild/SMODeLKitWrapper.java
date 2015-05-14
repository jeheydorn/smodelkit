package wrapper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.learner.SupervisedLearner;
import smodelkit.util.Pair;
import smodelkit.util.Range;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;

/**
 * This is a wrapper to allow me to use models from SMODeLKit as base-models in Weka.
 * 
 * If used in the Meka toolkit, this wrapper will only work to wrap a multi-class
 * classifier; it will not work for a multi-label or multi-target classifier.
 * @author joseph
 *
 */
public class SMODeLKitWrapper extends AbstractClassifier implements
		OptionHandler, WeightedInstancesHandler, Randomizable
{
	private static final long serialVersionUID = 1L;
	String modelName;
	String modelSettingsFile;
	SupervisedLearner model;
	int seed;

	public SMODeLKitWrapper()
	{
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception
	{
		Random rand = new Random(seed);
		// If I want to use settings files which reference other settings files, I will need
		// to set the directory in which MLSystemsManager looks for settings files. Currently
		// I do not do that.
		model = MLSystemsManager.createLearner(rand, modelName, modelSettingsFile);
		Matrix data = convertToSMODeLKitFormat(instances);
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
				
		model.train(inputs, labels);
	}
	
	/**
	 * Converts the given instances to the format I use in SMODeLKit. This assumes
	 * that there is only 1 class attribute, that it is stored as the first attribute,
	 * and that it is obtained by calling instances.classAttribute().
	 * @param instances
	 * @return An array of matrixes (datasets). The zero'th element is the inputs,
	 * and the 1st element is the labels.
	 */
	private static Matrix convertToSMODeLKitFormat(Instances instances)
	{		
		Matrix data = new Matrix();
		@SuppressWarnings("unchecked")
		Enumeration<Attribute> enumeration = instances.enumerateAttributes();
		List<Attribute> attributes = new ArrayList<Attribute>();
		// Store the class attribute last. This is the way SMODeLKit expects it.
		attributes.addAll(Collections.list(enumeration));
		attributes.add(instances.classAttribute());
		for (Attribute attr : attributes)
		{
			data.addEmptyColumn(attr.name());
			if (attr.isNominal())
			{
				for (int i : new Range(attr.numValues()))
				{
					data.addAttributeValue(data.cols() - 1, attr.value(i));
				}
			}
			else if (attr.isNumeric())
			{
			}
			else if (attr.isDate())
			{
				throw new UnsupportedOperationException("Date type attributes are not supported.");
			}
			else if (attr.isString())
			{
				throw new UnsupportedOperationException("String type attributes are not supported.");
			}
			else
			{
				throw new IllegalArgumentException("Unable to determine attribute type.");
			}
		}
		
		for (Instance instance : instances)
		{
			Vector newRow = convertToSMODeLKitFormat(instance);
			data.addRow(newRow);
		}
				
		data.setNumLabelColumns(1);
		
		return data;
	}

	private static Vector convertToSMODeLKitFormat(Instance instance)
	{		
		// Move the class value to be the last element of the result.
		double[] result = new double[instance.numAttributes()];
		for (int i : new Range(instance.numAttributes()))
		{
			int resultIndex = i;
			if (i == instance.classIndex())
				resultIndex = result.length - 1;
			else if (i > instance.classIndex())
				resultIndex--;
						
			result[resultIndex] = instance.value(i);
		}
				
		return new Vector(result, instance.weight());
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception
	{	
		Vector inputVec = convertToSMODeLKitFormat(instance);
		// Meka gives me the class value with the instance, but SMODeLKit does not expect
		// the class value, so I am removing it here. 
		Vector input = inputVec.subVector(0, inputVec.size() - 1);
		List<double[]> weights = model.predictOutputWeights(input);
		
		assert weights.size() == 1;
		
		// Convert the weights to the format expected by Meka.
		return weights.get(0);
	}

	@Override
	public void setSeed(int seed)
	{
		this.seed = seed;
	}

	@Override
	public int getSeed()
	{
		return seed;
	}

	@Override
	public Capabilities getCapabilities()
	{
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);

		return result;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions()
	{

		java.util.Vector<Option> newVector = new java.util.Vector<Option>(14);

		newVector.addElement(new Option(
				"\tThe name of a model in from the SMODeLKit.\n", "M", 1,
				"-M <model name>"));

		newVector.addElement(new Option(
				"\tThe location of the model's settings file.\n", "A", 1,
				"-F <settings file>"));

		return newVector.elements();

	}

	@Override
	public void setOptions(String[] options) throws Exception
	{
		modelName = Utils.getOption('M', options);
		modelSettingsFile = Utils.getOption('F', options);
		String seedStr = Utils.getOption('S', options);
		if (seedStr != null)
		{
			if (seedStr.toLowerCase().equals("random"))
				seed = new Random(System.currentTimeMillis()).nextInt();
			else
				seed = Integer.parseInt(seedStr);
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}
	
	public static void testPrivateMethods() throws Exception
	{	 	
		testOnIris();
		testOnAdult();
	}
	
	private static void testOnAdult() throws Exception
	{
		String arff = "../SMODeLKit/Datasets/mcc/adult.arff";
	 	Instances instances = loadInstancesForTest(arff);
	 	
	 	Matrix convertedData = convertToSMODeLKitFormat(instances);
	 	
	 	Matrix expected = new Matrix();
	 	expected.loadFromArffFile(arff);
	 	
	 	assertEquals(expected.rows(), convertedData.rows());
	 	assertEquals(expected.cols(), convertedData.cols());
	 	for (int r : new Range(expected.rows()))
	 	{
	 		for (int c : new Range(expected.cols()))
	 		{
	 			assertEquals(expected.row(r).get(c), convertedData.row(r).get(c), Double.MIN_VALUE);
	 		}
	 	}
	}
	
	private static void testOnIris() throws Exception
	{
	 	Instances instances = loadInstancesForTest("../SMODeLKit/Datasets/mcc/iris.arff");
	 	
	 	Matrix dataset = convertToSMODeLKitFormat(instances);
	 	Pair<Matrix> inputsAndLabels = dataset.splitInputsAndLabels();
	 	Matrix inputs = inputsAndLabels.getFirst();
	 	Matrix labels = inputsAndLabels.getSecond();
	 	
	 	
	 	
	 	assertEquals(4, inputs.cols());
	 	assertTrue(inputs.isContinuous(0));
	 	assertTrue(inputs.isContinuous(1));
	 	assertTrue(inputs.isContinuous(2));
	 	assertTrue(inputs.isContinuous(3));
	 	assertEquals(150, inputs.rows());
	 	assertEquals("5.1,3.5,1.4,0.2", inputs.rowToString(0));
	 	assertEquals("5.9,3,5.1,1.8", inputs.rowToString(inputs.row(inputs.rows() - 1)));

	 	assertEquals(1, labels.cols());
	 	assertFalse(labels.isContinuous(0));
	 	assertEquals(3, labels.getValueCount(0));
	 	assertEquals(150, labels.rows());
	 	assertEquals("Iris-setosa", labels.rowToString(0));
	 	assertEquals("Iris-virginica", labels.rowToString(labels.rows() - 1));
	 	assertEquals(2.0, labels.row(labels.rows() - 1).get(0), Double.MIN_VALUE);
	}
	
	private static Instances loadInstancesForTest(String arffFileName) throws Exception
	{
	 	Instances instances = Evaluation.loadDataset(new String[] {"-t", arffFileName});
	 	// This will parse the -c option in the relation name and move the class attribute to the correct column.
	 	MLUtils.prepareData(instances);
	 	// For some reason the above line sets the class index to 1, but it should be 0.
	 	instances.setClassIndex(0);
	 	return instances;
	}
	
	public static void main(String[] argv)
	{
		runClassifier(new ZeroR(), argv);
	}

}
