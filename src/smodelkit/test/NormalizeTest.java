package smodelkit.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static smodelkit.Vector.assertVectorEquals;

import org.junit.Test;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.filter.Filter;
import smodelkit.filter.Normalize;
import smodelkit.util.Pair;
import smodelkit.util.Range;

public class NormalizeTest
{	
	@Test
	public void filterAllInputsTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/iris_test.arff", 1);
		
		Filter filter = new Normalize();
		filter.initialize(inputsAndLabels[0], inputsAndLabels[1]);
			
		Matrix filteredInputs = filter.filterAllInputs(inputsAndLabels[0]);

		assertVectorEquals(Vector.create(-1.0, 1.0, -1.0, -1.0), filteredInputs.row(0), 0.0000001);
		assertVectorEquals(Vector.create(((7.0 - 5.1))/(7.3 - 5.1) * 2.0 - 1.0,
				(3.2 - 2.9)/(3.5 - 2.9) * 2.0 - 1.0,
				(4.7 - 1.4)/(6.3 - 1.4) * 2.0 - 1.0,
				(1.4 - 0.2)/(1.8 - 0.2) * 2.0 - 1.0), filteredInputs.row(1), 0);
		assertVectorEquals(Vector.create(1.0, -1.0, 1.0, 1.0), filteredInputs.row(2), 0.0000001);
		
	}
	
	@Test
	public void filterAllInputsWithUnknownsTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/iris_test_with_unknowns.arff", 1);
		
		Filter filter = new Normalize();
		filter.initialize(inputsAndLabels[0], inputsAndLabels[1]);
			
		Matrix filteredInputs = filter.filterAllInputs(inputsAndLabels[0]);

		assertVectorsEqualWithUnknowns(Vector.create(Vector.getUnknownValue(), 1.0, -1.0, -1.0), filteredInputs.row(0), 0.0000001);
		assertVectorEquals(Vector.create(-1,
				(3.2 - 2.9)/(3.5 - 2.9) * 2.0 - 1.0,
				1.0,
				(1.4 - 0.2)/(1.8 - 0.2) * 2.0 - 1.0), filteredInputs.row(1), 0);
		assertVectorsEqualWithUnknowns(Vector.create(1.0, -1.0, Vector.getUnknownValue(), 1.0), filteredInputs.row(2), 0.0000001);
				
	}
	
	private static void assertVectorsEqualWithUnknowns(Vector a1, Vector a2, double threshold)
	{
		assertEquals(a1.size(), a2.size());
		for (int i : new Range(a1.size()))
		{
			if (Vector.isUnknown(a1.get(i)) || Vector.isUnknown(a2.get(i)))
			{
				assertTrue(Vector.isUnknown(a1.get(i)));
				assertTrue(Vector.isUnknown(a2.get(i)));
			}
			else
			{
				assertEquals(a1.get(i), a2.get(i), threshold);
			}
		}
	}

	@Test
	public void filterAllLabelsTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/iris_test.arff", 1);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new Normalize();
		filter.initialize(inputs, labels);
		
		Matrix labelsF = filter.filterAllLabels(labels);

		assertVectorEquals(Vector.create(0.0), labelsF.row(0), 0);
		assertVectorEquals(Vector.create(1.0), labelsF.row(1), 0);
		assertVectorEquals(Vector.create(2.0), labelsF.row(2), 0);
		
		
	}

	@Test
	public void filterInputTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/iris_test.arff", 1);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new Normalize();
		filter.initialize(inputs, labels);
		
		assertVectorEquals(Vector.create(-1.0, 1.0, -1.0, -1.0), filter.filterInput(inputs.row(0)), 0);
		assertVectorEquals(Vector.create(((7.0 - 5.1))/(7.3 - 5.1) * 2.0 - 1.0,
				(3.2 - 2.9)/(3.5 - 2.9) * 2.0 - 1.0,
				(4.7 - 1.4)/(6.3 - 1.4) * 2.0 - 1.0,
				(1.4 - 0.2)/(1.8 - 0.2) * 2.0 - 1.0), filter.filterInput(inputs.row(1)), 0.0000001);
		assertVectorEquals(Vector.create(1.0, -1.0, 1.0, 1.0), filter.filterInput(inputs.row(2)), 0);

	}

	@Test
	public void unfilterLabelTest() throws Exception
	{
		Matrix[] inputsAndLabels = loadInputsAndLabels("Datasets/test/iris_test.arff", 1);
		Matrix inputs = inputsAndLabels[0];
		Matrix labels = inputsAndLabels[1];		
		Filter filter = new Normalize();
		filter.initialize(inputs, labels);
		
		Vector labelF = filter.filterAllLabels(labels).row(0);
		Vector labelUF = filter.unfilterLabel(labelF);
		assertVectorEquals(labels.row(0), labelUF, 0);

		labelF = filter.filterAllLabels(labels).row(1);
		labelUF = filter.unfilterLabel(labelF);
		assertVectorEquals(labels.row(1), labelUF, 0);
	}
	
	@Test
	public void unfilterAllLabelsNumericTest()
	{
		Matrix data = new Matrix();
		// A snippet from Datsets/mtr/edm.arff.
		data.loadFromArffString("% Electrical discharge machining, data 1, numeric\n" + 
				"%\n" + 
				"% For more info, see A. Karalic PhD\n" + 
				"\n" + 
				"@relation 'edm1: -c -2'\n" + 
				"\n" + 
				"@attribute ASM_A_MeanT numeric\n" + 
				"@attribute ASD_A_SDevT numeric\n" + 
				"@attribute BSM_B_MeanT numeric\n" + 
				"@attribute BSD_B_SDevT numeric\n" + 
				"@attribute CSM_C_MeanT numeric\n" + 
				"@attribute CSD_C_SDevT numeric\n" + 
				"@attribute ISM_I_MeanT numeric\n" + 
				"@attribute ISD_I_SDevT numeric\n" + 
				"@attribute ALM_A_MeanT numeric\n" + 
				"@attribute ALD_A_SDevT numeric\n" + 
				"@attribute BLM_B_MeanT numeric\n" + 
				"@attribute BLD_B_SDevT numeric\n" + 
				"@attribute CLM_C_MeanT numeric\n" + 
				"@attribute CLD_C_SDevT numeric\n" + 
				"@attribute ILM_I_MeanT numeric\n" + 
				"@attribute ILD_I_SDevT numeric\n" + 
				"@attribute DFlow numeric\n" + 
				"@attribute DGap numeric\n" + 
				"\n" + 
				"@data\n" + 
				"-4.86,0.04,0.33,0.13,5.83,0.15,0.97,0.03,-4.85,0.15,0.27,0.16,5.67,0.36,1.07,0.39,0,1\n" + 
				"-4.86,0.04,0.33,0.13,5.83,0.15,0.97,0.03,-4.85,0.15,0.27,0.16,5.67,0.36,1.07,0.39,0,1\n" + 
				"-4.86,0.04,0.33,0.13,5.83,0.15,0.97,0.03,-4.85,0.15,0.27,0.16,5.67,0.36,1.07,0.39,0,1\n" + 
				"-4.68,0.10,0.59,0.17,5.56,0.14,1.82,0.34,-4.75,0.13,0.43,0.19,5.73,0.21,1.38,0.41,1,0\n" + 
				"-4.68,0.10,0.59,0.17,5.56,0.14,1.82,0.34,-4.75,0.13,0.43,0.19,5.73,0.21,1.38,0.41,1,0\n" + 
				"-4.63,0.11,0.58,0.19,5.54,0.11,1.81,0.34,-4.73,0.14,0.42,0.19,5.72,0.20,1.41,0.40,1,0\n" + 
				"-4.81,0.07,0.58,0.28,5.58,0.23,1.38,0.36,-4.74,0.11,0.50,0.22,5.66,0.19,1.52,0.38,0,1\n" + 
				"-4.56,0.09,0.90,0.14,5.28,0.18,2.34,0.35,-4.62,0.11,0.75,0.23,5.45,0.23,2.06,0.40,0,0\n" + 
				"-4.52,0.09,1.09,0.04,5.12,0.00,2.33,0.29,-4.57,0.16,1.07,0.10,5.14,0.06,2.40,0.42,0,0\n" + 
				"-4.46,0.05,1.06,0.02,5.12,0.00,2.00,0.19,-4.54,0.15,1.07,0.09,5.14,0.06,2.38,0.42,-1,0\n" + 
				"-4.46,0.05,1.06,0.02,5.12,0.00,2.00,0.19,-4.54,0.15,1.07,0.09,5.14,0.06,2.38,0.42,-1,0\n" + 
				"-4.45,0.02,1.05,0.01,5.12,0.00,1.90,0.09,-4.53,0.15,1.08,0.07,5.12,0.01,2.38,0.40,-1,0\n" + 
				"-4.43,0.03,1.03,0.03,5.12,0.00,1.84,0.10,-4.52,0.14,1.07,0.07,5.12,0.01,2.33,0.41,-1,0\n" + 
				"-4.47,0.07,1.05,0.04,5.12,0.00,1.82,0.07,-4.51,0.14,1.07,0.07,5.12,0.01,2.28,0.40,-1,0\n" + 
				"-4.57,0.08,1.11,0.03,5.18,0.08,2.30,0.44,-4.50,0.14,1.07,0.07,5.14,0.05,2.19,0.37,0,-1\n" + 
				"");
		
		Filter filter = new Normalize();
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix labels = inputsAndLabels.getSecond();
		filter.initialize(inputsAndLabels.getFirst(), labels);
		Matrix labelsF = filter.filterAllLabels(labels);
		
		for (int i : new Range(labels.rows()))
		{
			assertVectorEquals(labels.row(i), filter.unfilterLabel(labelsF.row(i)), 0.0);
		}

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
