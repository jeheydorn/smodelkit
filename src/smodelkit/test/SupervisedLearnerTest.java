package smodelkit.test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;

import org.junit.Test;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.test.learners.MockLearner;
import smodelkit.util.Pair;

public class SupervisedLearnerTest
{

	@Test
	public void canImplicitlyHandleUnknownOutputsTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException
	{	
		try
		{
			new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/mdc/bridges_original.arff -E training -M top-n 1 -F mean_mode -R 0".split(" "), null);
		}
		catch (IllegalArgumentException e)
		{
			String expectedMessage = "NeuralNet cannot handle unknown outputs.";
			if (!e.getMessage().equals(expectedMessage))
			{
				e.printStackTrace();
				assertEquals(expectedMessage, e.getMessage());
			}
			return;
		}
		fail();
	}

	@Test
	public void predictOutputWeightsDefaultImplementationTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'test -c -2 '\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{a0, a1}\n" + 
				"@ATTRIBUTE x2	{b0, b1}\n" + 
				"@ATTRIBUTE class   {c0, c1, c2}\n" + 
				"\n" + 
				"@DATA\n" + 
				"a0,b0,c2\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		MockLearner learner = new MockLearner(Arrays.asList(new double[] {0, 2}));
		learner.train(inputsAndLabels.getFirst(), inputsAndLabels.getSecond());
		List<double[]> weights = learner.predictOutputWeights(inputsAndLabels.getFirst().row(0));
		assertEquals(2, weights.size());
		assertArrayEquals(new double[] {1.0, 0.0}, weights.get(0), 0.0000000001);
		assertArrayEquals(new double[] {0.0, 0.0, 1.0}, weights.get(1), 0.0000000001);
	}
	
}
