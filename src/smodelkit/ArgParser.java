package smodelkit;

import java.util.Arrays;
import java.util.List;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;

class ArgParser
{
	@Parameter(names = {"-A", "--dataset"}, description = "arff file or .names and .data files.", required = true, variableArity=true)
	List<String> dataset;
	
	@Parameter(names = {"-L", "--learner"}, description = "learning algorithm and settings file. The learning algorithm"
			+ " can be a nickname (see MLSystemsManager.createLearner for a list of nicknames) or the canonical class"
			+ " name of a learner.", required = false, variableArity=true)
	List<String> learner;
	
	@Parameter(names = {"-D", "--deserialize"}, description = "File name of model to deserialize. This should be " +
			"relative to the \"models\" directory.")
	String deserializeFileName;
	
	@Parameter(names = {"-E", "--evaluation"}, description = "evaluation method. This can be one of: \n" +
		"\t\t-E training\n" +                
		"\t\t-E unsupervised\n" +                
		"\t\t-E static [TestARFF_File]\n" +
		"\t\t-E random [PercentageForTesting]\n" +
		"\t\t-E cross [numOfFolds]\n", 
		variableArity = true)	
	List<String> evaluation;
		
	@Parameter(names = {"-R", "--seed"}, description = "Random seed to use.")
	String seedStr = null;
	
	@Parameter(names = {"-I", "--ignore"}, description = "Attribute columns to ignore in the dataset. If numbers are given," +
			"columns wil be removed by index starting at zero. Index -1 is the last column." +
			"If column names are given, the columns with those names will be removed.", variableArity = true)
	List<String> ignoredColumns;
			
	@Parameter(names = {"-U", "--labels_count"}, description = "The number of columns from the right side of the dataset to use" +
			" as labels. This is not necessary when -C is used.")
	Integer numLabelColumns = null;
	
	// 
	@Parameter(names = {"-C", "--label_columns"}, description = "Names of columns that will be used as labels.",
			variableArity = true)
	List<String> labelColumnNames;
	
	@Parameter (names = {"-S", "--column_order"}, description = "A list of indexes, starting with 0, or " +
			"column names, which indicate the order in which label columns should be placed. This only has" +
			" effect when multiple output labels are used with -U.", variableArity = true)
	List<Integer> labelColumnOrder;
	
	@Parameter(names = {"-M", "--evaluator"}, 
			description = "The evaluator to use when evaluating the learner on the test set. Multiple can"
					+ " be specified, and each can have arguments. If multiple evaluators are specified, they"
					+ " must be separated by the token \"end\" without double quoates.", 
			variableArity = true)
	List<String> evaluators;
	
	@Parameter(names = {"-F", "--fill_unknowns"}, description = "How to fill or remove unknown values in the dataset. "
			+ "See MLSystemsManager.fillOrRemoveUnknownData for a list of possible values. Another way to fill unknowns"
			+ " is to use smodelkit.filter.MeanModeUnknownFiller in the \"filter\" part of a model's settings. The difference "
			+ "is that this method uses the entire dataset to fill unknowns, while the filter method uses only the model's"
			+ " training data. This method also saves a little memory because it does not keep a copy of the instances "
			+ "before unknowns were filled. The filter only copies an instance if it contains an unknown value.", 
			variableArity = true)
	List<String> unknownFiller = Arrays.asList("none");
	
    @Parameter(names = {"-J", "--threads"}, description = "Number of threads to use. Setting this to more than the "
    		+ "number of threads on the system will have no effecrt. This number is only a guidline for the number"
    		+ " of worker threads to create.")
    Integer maxThreads = null;

    @Parameter(names = {"--rows"}, description = "Number of rows from the data set to use. This is usefull" +
    		" to see how a model does with less data. The first n rows will be used, where n is this paramter.")
    Integer numRows = null;
    
    @Parameter(names = {"--oversample_training_data"}, description = "Oversample class in the trainin data unitl classes"
    		+ "are balanced.")
    boolean oversample = false;
    
    @Parameter(names = {"--print_percent_unique_test_labels"}, description = "Find the number of label in the test which"
    		+ " are not in the training set for whichever evaluation technique is used")
    boolean printPercentUniqueTestLabels;
   
    @Parameter(names = {"--print_metadata_only"}, description = "Do not train or evaluate: only print metadata.")
    boolean printMetadataOnly;
    
    @Parameter(names = {"--include_training_data_evaluations"}, description = "Evaluate on the training data as well as any"
    		+ " test data specified with -E.")
    boolean includeTrainingDataEvaluations;
    
    @Parameter(names = {"--float"}, description = "Causes all vectors to be loaded using floats instead of doubles. This will save"
    		+ " some memory but will slow down a learner unless it is written to use floats.")
    boolean useFloats = false;
    

    
	public static ArgParser parse(String[] args)
	{
		ArgParser parser = new ArgParser();
		JCommander jc = new JCommander(parser);
		try
		{
			jc.parse(args);
		}
		catch(ParameterException ex)
		{
			System.out.println(ex.getMessage());
			jc.setProgramName("MLSystemManager");
			jc.usage();
			System.exit(1);
		}
		return parser;
	}
	
	private ArgParser()
	{
		
	}

}

