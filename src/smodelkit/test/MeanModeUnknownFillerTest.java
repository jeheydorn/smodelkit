package smodelkit.test;

import static smodelkit.Vector.assertVectorEquals;

import org.junit.Test;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.filter.MeanModeUnknownFiller;
import smodelkit.util.Pair;

public class MeanModeUnknownFillerTest
{

	@Test
	public void fillAllEmptyDataTest() throws Exception
	{
		Matrix dataset = loadDataset();
		MeanModeUnknownFiller filler = new MeanModeUnknownFiller();
		Pair<Matrix> inputsAndLabels = dataset.splitInputsAndLabels();
		filler.initialize(inputsAndLabels.getFirst(), inputsAndLabels.getSecond());
		Matrix actual = filler.filterAllInputs(inputsAndLabels.getFirst());
		
		assertVectorEquals(Vector.create(7.5, 3.5, 1.4, 5.0, 0.2), actual.row(0), 0);
		assertVectorEquals(Vector.create(7.0, 3.2, 4.7, 0.0, 1.4), actual.row(1), 0);
		assertVectorEquals(Vector.create(7.3, 2.9, 3.05, 0.0, 1.8), actual.row(2), 0);
		assertVectorEquals(Vector.create(8.2, 3.3, 3.05, 0.0, 1.133333333), actual.row(3), 0.000001);
	}	

	private Matrix loadDataset()
	{
		String mixedUnknownsDataset = "@RELATION iris\n" + 
				"\n" + 
				"@ATTRIBUTE a1	Continuous\n" + 
				"@ATTRIBUTE a2 	Continuous\n" + 
				"@ATTRIBUTE a3 	Continuous\n" + 
				"@ATTRIBUTE a4	{type0, type1, type2, type3, type4, type5}\n" + 
				"@ATTRIBUTE a5	Continuous\n" + 
				"@ATTRIBUTE class 	{label0,label1,label2}\n" + 
				"\n" + 
				"@DATA\n" + 
				"?,3.5,1.4,type5,0.2,label0\n" + 
				"7.0,3.2,4.7,type0,1.4,label2\n" + 
				"7.3,2.9,?,type0,1.8,label0\n" + 
				"8.2,3.3,?,?,?,label1";

		Matrix data = new Matrix();		
		data.loadFromArffString(mixedUnknownsDataset);
		return data;
	}

}
