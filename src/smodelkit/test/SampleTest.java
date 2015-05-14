package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.util.Random;

import org.junit.Test;

import smodelkit.Matrix;
import smodelkit.Sample;
import smodelkit.util.Pair;

public class SampleTest
{

	@Test
	public void sampleWithReplacement100PercentTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'simple_nominal -c -2'\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE x3   {up, down}\n" + 
				"@ATTRIBUTE class1	{pig,sheep}\n" + 
				"@ATTRIBUTE class2	{tasty,not-tasty}\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,8.88,up,pig,tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"red,0.112,up,pig,not-tasty\n" + 
				"\n" + 
				"\n" + 
				"\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Matrix[] actual = Sample.sampleWithReplacement(new Random(), inputs, labels, 1.0);
		assertEquals(4, actual[0].rows());
		assertEquals(4, actual[1].rows());
		assertEquals(3, actual[0].cols());
		assertEquals(2, actual[1].cols());
	}

	@Test
	public void sampleWithReplacement0PercentTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'simple_nominal -c -2'\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE x3   {up, down}\n" + 
				"@ATTRIBUTE class1	{pig,sheep}\n" + 
				"@ATTRIBUTE class2	{tasty,not-tasty}\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,8.88,up,pig,tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"red,0.112,up,pig,not-tasty\n" + 
				"\n" + 
				"\n" + 
				"\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Matrix[] actual = Sample.sampleWithReplacement(new Random(), inputs, labels, 0.0);
		assertEquals(0, actual[0].rows());
		assertEquals(0, actual[1].rows());
		assertEquals(3, actual[0].cols());
		assertEquals(2, actual[1].cols());
	}

	@Test
	public void sampleWithReplacement20PercentTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'simple_nominal -c -2'\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE x3   {up, down}\n" + 
				"@ATTRIBUTE class1	{pig,sheep}\n" + 
				"@ATTRIBUTE class2	{tasty,not-tasty}\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,8.88,up,pig,tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"red,0.112,up,pig,not-tasty\n" + 
				"\n" + 
				"\n" + 
				"\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Matrix[] actual = Sample.sampleWithReplacement(new Random(), inputs, labels, 0.2);
		assertEquals(1, actual[0].rows());
		assertEquals(1, actual[1].rows());
		assertEquals(3, actual[0].cols());
		assertEquals(2, actual[1].cols());
	}

}
