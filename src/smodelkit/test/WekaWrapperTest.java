package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import org.junit.Test;

import smodelkit.Evaluation;
import smodelkit.MLSystemsManager;
import smodelkit.evaluator.TopN;

public class WekaWrapperTest
{

	@Test
	public void test() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException
	{
		// An MDC dataset with unknowns.
		Evaluation evaluation = new MLSystemsManager().run("-L ic model_settings/ic_SMO.json -A Datasets/mdc/bridges.arff -E cross 5 -M top-n 1 -R 0".split(" "), null);
		assertEquals(0.2, evaluation.getScores(TopN.class).get(0), 0.000000001);

		
		evaluation = new MLSystemsManager().run("-L weka model_settings/weka_SMO.json -A Datasets/mcc/iris.arff -E cross 5 -M top-n 1 -R 1".split(" "), null);
		assertEquals(0.96, evaluation.getScores(TopN.class).get(0), 0.000000001);
	}

}
