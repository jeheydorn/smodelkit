package smodelkit.test;

import static smodelkit.Vector.assertVectorEquals;

import org.junit.Test;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.filter.Filter;
import smodelkit.filter.NominalToCategorical;
import smodelkit.filter.Normalize;
import smodelkit.util.Pair;

public class NominalToCategoricalTest
{
	
	@Test
	public void filterAllInputsTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/small_nominal.arff", 2);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new NominalToCategorical(true);
		filter.initialize(inputs, labels);

		Matrix filteredInputs = filter.filterAllInputs(inputs);

		assertVectorEquals(new VectorDouble(1, 0, 0, 8.88), filteredInputs.row(0), 0);
		assertVectorEquals(new VectorDouble(0, 0, 1, 999.1), filteredInputs.row(1), 0);
	}

	@Test
	public void filterAllLabels1BinaryTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/small_nominal.arff", 2);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new NominalToCategorical(true);
		filter.initialize(inputs, labels);
		
		Matrix labelsF = filter.filterAllLabels(labels);
		
		assertVectorEquals(new VectorDouble(0, 1, 0, 0), labelsF.row(0), 0);
		assertVectorEquals(new VectorDouble(1, 0, 0, 1), labelsF.row(1), 0);
		
	}

	@Test
	public void filterAllLabels2BinaryTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/small_nominal.arff", 2);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new NominalToCategorical(false);
		filter.initialize(inputs, labels);
		
		Matrix labelsF = filter.filterAllLabels(labels);
		
		assertVectorEquals(new VectorDouble(1, 0, 1, 0, 0), labelsF.row(0), 0);
		assertVectorEquals(new VectorDouble(0, 1, 0, 0, 1), labelsF.row(1), 0);
		
	}

	@Test
	public void filterInputTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/small_nominal.arff", 2);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new NominalToCategorical(true);
		filter.initialize(inputs, labels);

		assertVectorEquals(new VectorDouble(1, 0, 0, 8.88), filter.filterInput(inputs.row(0)), 0);
		assertVectorEquals(new VectorDouble(0, 0, 1, 999.1), filter.filterInput(inputs.row(1)), 0);
	}

	@Test
	public void unfilterLabelTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/small_nominal.arff", 2);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new NominalToCategorical(true);
		filter.initialize(inputs, labels);

		Vector labelF = filter.filterAllLabels(labels).row(0);
		Vector labelUF = filter.unfilterLabel(labelF);
		assertVectorEquals(labels.row(0), labelUF, 0);

		labelF = filter.filterAllLabels(labels).row(1);
		labelUF = filter.unfilterLabel(labelF);
		assertVectorEquals(labels.row(1), labelUF, 0);
	}

	@Test
	public void unfilterContinuousAndBinaryLabelWith1BinaryTest() throws Exception
	{
		Matrix data = new Matrix();
		data.loadFromArffString(
				"@RELATION 'simple_nominal: -c -2 '\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE class1	{pig,sheep}\n" + 
				"@ATTRIBUTE class2	real\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,pig,8.88\n" + 
				"green,sheep,999.1,\n" + 
				"\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Filter filter = new NominalToCategorical(true);
		filter.initialize(inputs, labels);

		Vector labelF = filter.filterAllLabels(labels).row(0);
		// Simulate a probability in class1. This way I know that binary nominal values are not mistaken
		// for continuous values.
		labelF.set(0, 0.4999);
		Vector labelUF = filter.unfilterLabel(labelF);
		assertVectorEquals(labels.row(0), labelUF, 0);

		labelF = filter.filterAllLabels(labels).row(1);
		// Simulate a probability in class1. This way I know that binary nominal values are not mistaken
		// for continuous values.
		labelF.set(0, 0.7);
		labelUF = filter.unfilterLabel(labelF);
		assertVectorEquals(labels.row(1), labelUF, 0);
	}
	
	@Test
	public void unfilterContinuousAndBinaryLabelWith2BinaryTest() throws Exception
	{
		Matrix data = new Matrix();
		data.loadFromArffString(
				"@RELATION 'simple_nominal: -c -2 '\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE class1	{pig,sheep}\n" + 
				"@ATTRIBUTE class2	real\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,pig,8.88\n" + 
				"green,sheep,999.1,\n" + 
				"\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Filter filter = new NominalToCategorical(false);
		filter.initialize(inputs, labels);

		Vector labelF = filter.filterAllLabels(labels).row(0);
		// Simulate a probability in class1. This way I know that binary nominal values are not mistaken
		// for continuous values.
		labelF.set(0, 1.0 - 0.4999);
		labelF.set(1, 0.4999);
		Vector labelUF = filter.unfilterLabel(labelF);
		assertVectorEquals(labels.row(0), labelUF, 0);

		labelF = filter.filterAllLabels(labels).row(1);
		// Simulate a probability in class1. This way I know that binary nominal values are not mistaken
		// for continuous values.
		labelF.set(0, 1.0 - 0.7);
		labelF.set(1, 0.7);
		labelUF = filter.unfilterLabel(labelF);
		assertVectorEquals(labels.row(1), labelUF, 0);
	}


	@Test
	public void unfilterAllLabelsTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/small_nominal_with_unknowns.arff", 2);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new NominalToCategorical(true);
		filter.initialize(inputs, labels);

		Matrix labelsF = filter.filterAllLabels(labels);
			
		for (int i = 0; i < labels.rows(); i++)
		{
			assertVectorEquals(labels.row(i), filter.unfilterLabel(labelsF.row(i)), 0);
		}
	}
	
	/**
	 * Tests nesting a NominaltoCategorical filter within a Normalize filter whith a dataset
	 * split into training and test data.
	 * @throws Exception
	 */
	@Test
	public void nestedTest() throws Exception
	{
		int labelCols = 2;
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/small_nominal.arff", labelCols);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];
		Filter filter = new NominalToCategorical(true);
		Normalize normalize = new Normalize();
		normalize.setInnerFilter(filter);
		filter = normalize;
		filter.initialize(inputs, labels);
		
		Matrix inputsF = filter.filterAllInputs(inputs);
		assertVectorEquals(new VectorDouble(1, 0, 0, -1), inputsF.row(0), 0);

		assertVectorEquals(new VectorDouble(0, 0, 1, 1.0), inputsF.row(1), 0);
		
		Matrix labelsF = filter.filterAllLabels(labels);
		assertVectorEquals(new VectorDouble(1, 0, 0, 1), labelsF.row(1), 0);

	}

	/**
	 * Tests nesting a NominaltoCategorical filter within a Normalize filter whith a dataset
	 * split into training and test data.
	 * @throws Exception
	 */
	@Test
	public void nestedTestWithNumericLabel() throws Exception
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	numeric\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,8.88,100\n" + 
				"green,999.1,12.2\n" + 
				"");
		
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Filter filter = new NominalToCategorical(true);
		Normalize normalize = new Normalize();
		normalize.setInnerFilter(filter);
		filter = normalize;
		filter.initialize(inputs, labels);
		
		Matrix inputsF = filter.filterAllInputs(inputs);
		assertVectorEquals(new VectorDouble(1, 0, 0, -1), inputsF.row(0), 0);

		assertVectorEquals(new VectorDouble(0, 0, 1, 1.0), inputsF.row(1), 0);
		
		Matrix labelsF = filter.filterAllLabels(labels);
		assertVectorEquals(new VectorDouble(1), labelsF.row(0), 0);
		assertVectorEquals(new VectorDouble(-1), labelsF.row(1), 0);

	}

	private Matrix[] loadInputsAndLabels(String fileName, int numOutputs) throws Exception
	{
		Matrix data = new Matrix();		
		data.loadFromArffFile(fileName);
		
		
		Matrix inputs = new Matrix(data, 0, 0, data.rows(),
				data.cols() - numOutputs);
		Matrix labels = new Matrix(data, 0, data.cols() - numOutputs, data.rows(), numOutputs);
		
		Matrix[] result = {inputs, labels};

		return result;
	}

}
