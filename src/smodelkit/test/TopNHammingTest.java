package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.evaluator.Evaluator;
import smodelkit.evaluator.TopNHamming;
import smodelkit.test.learners.ScoredListMockLearner;
import smodelkit.util.Pair;

public class TopNHammingTest
{

	@Test
	public void test()
	{
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION 'simple_nominal -c -3'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a,b,c}\n" + 
					"@ATTRIBUTE x2	real\n" + 
					"@ATTRIBUTE class1	{a,b}\n" + 
					"@ATTRIBUTE class2	{a,b,c}\n" + 
					"@ATTRIBUTE class3	{a,b,c,d}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a,0.1,a,a,a\n" + 
					"");
			Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
			Matrix inputs = inputsAndLabels.getFirst();
			Matrix labels = inputsAndLabels.getSecond();
			
			List<Vector> pred1 = Arrays.asList(new VectorDouble(new double[]{0.0, 0.0, 0.0}, 0.9));
			ScoredListMockLearner learner = new ScoredListMockLearner(Arrays.asList(pred1));
			TopNHamming evaluator = new TopNHamming();
			evaluator.configure(Arrays.asList(1, 2, 3));
			assertEquals(Arrays.asList(1.0, 1.0, 1.0), Evaluator.runEvaluators(inputs, labels, learner, true, Arrays.asList(evaluator)).getScores(evaluator.getClass()));
		}
		
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION 'simple_nominal -c -3'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a,b,c}\n" + 
					"@ATTRIBUTE x2	real\n" + 
					"@ATTRIBUTE class1	{a,b}\n" + 
					"@ATTRIBUTE class2	{a,b,c}\n" + 
					"@ATTRIBUTE class3	{a,b,c,d}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a,0.1,a,a,a\n" + 
					"");
			Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
			Matrix inputs = inputsAndLabels.getFirst();
			Matrix labels = inputsAndLabels.getSecond();
			
			List<Vector> pred1 = Arrays.asList(
					new VectorDouble(new double[]{1.0, 1.0, 0.0}, 0.9),
					new VectorDouble(new double[]{0.0, 0.0, 1.0}, 0.8),
					new VectorDouble(new double[]{0.0, 0.0, 0.0}, 0.6));
			ScoredListMockLearner learner = new ScoredListMockLearner(Arrays.asList(pred1));
			TopNHamming evaluator = new TopNHamming();
			evaluator.configure(Arrays.asList(1, 2, 3));
			List<Double> scores = Evaluator.runEvaluators(inputs, labels, learner, true, Arrays.asList(evaluator)).getScores(evaluator.getClass());
			assertEquals(0.333333333333333, scores.get(0), 0.000000001);
			assertEquals(0.666666666666666, scores.get(1), 0.000000001);
			assertEquals(1.0, scores.get(2), 0.000000001);
		}
	
		// Test with all predictions at least partially wrong.
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION 'simple_nominal -c -3'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a,b,c}\n" + 
					"@ATTRIBUTE x2	real\n" + 
					"@ATTRIBUTE class1	{a,b}\n" + 
					"@ATTRIBUTE class2	{a,b,c}\n" + 
					"@ATTRIBUTE class3	{a,b,c,d}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a,0.1,a,a,a\n" + 
					"");
			Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
			Matrix inputs = inputsAndLabels.getFirst();
			Matrix labels = inputsAndLabels.getSecond();
			
			List<Vector> pred1 = Arrays.asList(
					new VectorDouble(new double[]{1.0, 1.0, 1.0}, 0.9),
					new VectorDouble(new double[]{0.0, 1.0, 1.0}, 0.8),
					new VectorDouble(new double[]{1.0, 1.0, 1.0}, 0.6));
			ScoredListMockLearner learner = new ScoredListMockLearner(Arrays.asList(pred1));
			TopNHamming evaluator = new TopNHamming();
			evaluator.configure(Arrays.asList(1, 2, 3));
			List<Double> scores = Evaluator.runEvaluators(inputs, labels, learner, true, Arrays.asList(evaluator)).getScores(evaluator.getClass());
			assertEquals(0.0, scores.get(0), 0.000000001);
			assertEquals(0.33333333333333, scores.get(1), 0.000000001);
			assertEquals(0.33333333333333, scores.get(2), 0.000000001);
		}
		
		// Test with values of n in reverse order.
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION 'simple_nominal -c -3'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a,b,c}\n" + 
					"@ATTRIBUTE x2	real\n" + 
					"@ATTRIBUTE class1	{a,b}\n" + 
					"@ATTRIBUTE class2	{a,b,c}\n" + 
					"@ATTRIBUTE class3	{a,b,c,d}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a,0.1,a,a,a\n" + 
					"");
			Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
			Matrix inputs = inputsAndLabels.getFirst();
			Matrix labels = inputsAndLabels.getSecond();
			
			List<Vector> pred1 = Arrays.asList(
					new VectorDouble(new double[]{1.0, 1.0, 0.0}, 0.9),
					new VectorDouble(new double[]{0.0, 0.0, 1.0}, 0.8),
					new VectorDouble(new double[]{0.0, 0.0, 0.0}, 0.6));
			ScoredListMockLearner learner = new ScoredListMockLearner(Arrays.asList(pred1));
			TopNHamming evaluator = new TopNHamming();
			evaluator.configure(Arrays.asList(3, 2, 1));
			List<Double> scores = Evaluator.runEvaluators(inputs, labels, learner, true, Arrays.asList(evaluator)).getScores(evaluator.getClass());
			assertEquals(1.0, scores.get(0), 0.000000001);
			assertEquals(0.666666666666666, scores.get(1), 0.000000001);
			assertEquals(0.333333333333333, scores.get(2), 0.000000001);
		}
	
		// Make sure an extra prediction (> n) is ignored.
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION 'simple_nominal -c -3'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a,b,c}\n" + 
					"@ATTRIBUTE x2	real\n" + 
					"@ATTRIBUTE class1	{a,b}\n" + 
					"@ATTRIBUTE class2	{a,b,c}\n" + 
					"@ATTRIBUTE class3	{a,b,c,d}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a,0.1,a,a,a\n" + 
					"");
			Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
			Matrix inputs = inputsAndLabels.getFirst();
			Matrix labels = inputsAndLabels.getSecond();
			
			List<Vector> pred1 = Arrays.asList(
					new VectorDouble(new double[]{1.0, 1.0, 1.0}, 0.9),
					new VectorDouble(new double[]{0.0, 1.0, 1.0}, 0.8),
					new VectorDouble(new double[]{1.0, 1.0, 1.0}, 0.6),
					new VectorDouble(new double[]{0.0, 0.0, 0.0}, 0.4));
			ScoredListMockLearner learner = new ScoredListMockLearner(Arrays.asList(pred1));
			TopNHamming evaluator = new TopNHamming();
			evaluator.configure(Arrays.asList(3));
			List<Double> scores = Evaluator.runEvaluators(inputs, labels, learner, true, Arrays.asList(evaluator)).getScores(evaluator.getClass());
			assertEquals(0.33333333333333, scores.get(0), 0.000000001);
		}
		
		// Test with multiple rows.
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION 'simple_nominal -c -3'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a,b,c}\n" + 
					"@ATTRIBUTE x2	real\n" + 
					"@ATTRIBUTE class1	{a,b}\n" + 
					"@ATTRIBUTE class2	{a,b,c}\n" + 
					"@ATTRIBUTE class3	{a,b,c,d}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a,0.1,b,b,b\n" + 
					"b,0.2,b,b,b\n" + 
					"a,3.1,b,c,d\n" + 
					"");

			Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
			Matrix inputs = inputsAndLabels.getFirst();
			Matrix labels = inputsAndLabels.getSecond();
			
			// n=1: 2/3, n=2: 2/3, n=3, 2/3
			List<Vector> pred1 = Arrays.asList(
					new VectorDouble(new double[]{1.0, 1.0, 0.0}, 0.9),
					new VectorDouble(new double[]{0.0, 0.0, 1.0}, 0.8),
					new VectorDouble(new double[]{0.0, 0.0, 0.0}, 0.6));
			// n=1: 0, n=2: 1/3, n=3, 1
			List<Vector> pred2 = Arrays.asList(
					new VectorDouble(new double[]{0.0, 0.0, 0.0}, 0.9),
					new VectorDouble(new double[]{0.0, 0.0, 1.0}, 0.8),
					new VectorDouble(new double[]{1.0, 1.0, 1.0}, 0.6));
			// n=1: 0, n=2: 1, n=3, 1
			List<Vector> pred3 = Arrays.asList(
					new VectorDouble(new double[]{0.0, 1.0, 2.0}, 0.9),
					new VectorDouble(new double[]{1.0, 2.0, 3.0}, 0.8),
					new VectorDouble(new double[]{0.0, 0.0, 0.0}, 0.6));
			ScoredListMockLearner learner = new ScoredListMockLearner(Arrays.asList(pred1, pred2, pred3));
			TopNHamming evaluator = new TopNHamming();
			evaluator.configure(Arrays.asList(1, 2, 3));
			List<Double> scores = Evaluator.runEvaluators(inputs, labels, learner, true, Arrays.asList(evaluator)).getScores(evaluator.getClass());
			assertEquals((2.0/3.0)/3.0, scores.get(0), 0.000000001);
			assertEquals(((2.0/3.0) + (1.0/3.0) + 1)/3.0, scores.get(1), 0.000000001);
			assertEquals(((2.0/3.0) + 1 + 1)/3.0, scores.get(2), 0.000000001);
		}
		
		
	}


}
