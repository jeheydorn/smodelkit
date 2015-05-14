package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import org.junit.Test;

import smodelkit.Evaluation;
import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.evaluator.TopN;

/**
 * If I refactor MLSystemsManager, these tests will help to tell me whether it functions the same afterward.
 * @author jeheydorn
 *
 */
public class MLSystemsManagerTest 
{

	@Test
	public void randomTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		Evaluation evaluation = new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/mcc/iris.arff -E random 0.1 -M top-n 1 -R 0".split(" "), null);
		assertEquals(0.9333333333333333, evaluation.getScores(TopN.class).get(0), 0.000000001);
		
	}

	@Test
	public void trainingTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		Evaluation evaluation = new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/mcc/iris.arff -E training -M top-n 1 -R 0".split(" "), null);
		assertEquals(0.98, evaluation.getScores(TopN.class).get(0), 0.000000001);
		
	}

	@Test
	public void staticTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		{
			Evaluation evaluation = new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/mcc/iris.arff -E static Datasets/mcc/iris.arff -M top-n 1 -R 0".split(" "), null);
			assertEquals(0.98, evaluation.getScores(TopN.class).get(0), 0.000000001);
		}
		{
			// Test where training and test sets are not exactly the same, and so static evaluation would not
			// give the same result as training evaluation.
			Evaluation evaluation = new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/mdc/synthetic/continuous_2out_4class.arff -E static Datasets/test/continuous_2out_4class_static_test.arff -M top-n 1 --rows 300 -R 0".split(" "), null);
			assertEquals(0.59375, evaluation.getScores(TopN.class).get(0), 0.000000001);
		}
	}

	@Test
	public void crossTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		Evaluation evaluation1 = new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/mcc/iris.arff -E cross 10 1 --threads 1 -M top-n 1 -R 0".split(" "), null);
		assertEquals(0.97333, evaluation1.getScores(TopN.class).get(0), 0.00001);

		// Make sure cross validation is deterministic when it is run on multiple threads.
		Evaluation evaluation2 = new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/mcc/iris.arff -E cross 10 1 --threads 8 -M top-n 1 -R 0".split(" "), null);
		assertEquals(0.97333, evaluation2.getScores(TopN.class).get(0), 0.00001);
	}
	
	@Test
	public void privateMethodTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		MLSystemsManager.testPrivateMethods();
	}
	
	@Test
	public void findPercentUniqueTestLabelsTest()
	{
		{
			Matrix trainLabels = new Matrix();
			trainLabels.loadFromArffString("@RELATION 'test -c -2'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"@ATTRIBUTE class2   {a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a, b\n" + 
					"a, c, b");
			Matrix testLabels = new Matrix();
			testLabels.loadFromArffString("@RELATION 'test -c -2'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"@ATTRIBUTE class2   {a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a, b\n" + 
					"a, c, b");
			assertEquals(0, MLSystemsManager.findPercentUniqueTestLabels(trainLabels, testLabels), 0.0000001);
		}

		{
			Matrix trainLabels = new Matrix();
			trainLabels.loadFromArffString("@RELATION 'test -c -2'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"@ATTRIBUTE class2   {a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a, b\n" + 
					"a, c, b");
			Matrix testLabels = new Matrix();
			testLabels.loadFromArffString("@RELATION 'test -c -2'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"@ATTRIBUTE class2   {a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a, a\n" + 
					"a, c, b");
			assertEquals(0.5, MLSystemsManager.findPercentUniqueTestLabels(trainLabels, testLabels), 0.0000001);
		}

		{
			Matrix trainLabels = new Matrix();
			trainLabels.loadFromArffString("@RELATION 'test -c -2'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"@ATTRIBUTE class2   {a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a, b\n" + 
					"a, c, b");
			Matrix testLabels = new Matrix();
			testLabels.loadFromArffString("@RELATION 'test -c -2'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"@ATTRIBUTE class2   {a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a, a\n" + 
					"a, c, a");
			assertEquals(1.0, MLSystemsManager.findPercentUniqueTestLabels(trainLabels, testLabels), 0.0000001);
		}

		// Test with only 1 output column.
		{
			Matrix trainLabels = new Matrix();
			trainLabels.loadFromArffString("@RELATION 'test -c -1'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a\n" + 
					"a, c");
			Matrix testLabels = new Matrix();
			testLabels.loadFromArffString("@RELATION 'test -c -1'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a\n" + 
					"a, c");
			assertEquals(0.0, MLSystemsManager.findPercentUniqueTestLabels(trainLabels, testLabels), 0.0000001);
		}

		{
			Matrix trainLabels = new Matrix();
			trainLabels.loadFromArffString("@RELATION 'test -c -1'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, a\n" + 
					"a, c");
			Matrix testLabels = new Matrix();
			testLabels.loadFromArffString("@RELATION 'test -c -1'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE class1	{a, b, c}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, b\n" + 
					"a, c");
			assertEquals(0.5, MLSystemsManager.findPercentUniqueTestLabels(trainLabels, testLabels), 0.0000001);
		}
}

}
