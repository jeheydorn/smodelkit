package smodelkit.util;

import static java.lang.System.out;

import java.util.HashSet;
import java.util.Set;

import smodelkit.MLSystemsManager;
import smodelkit.MetadataPrinter;
import smodelkit.learner.NeuralNet;

/**
 * A simple console logger.
 *
 */
public class Logger
{
	private static Set<String> loggingClassNames;
	private static String tabs;
	private static char lastChar;
	
	static
	{
		loggingClassNames = new HashSet<String>();
		//loggingClassNames.add(KNN.class.getCanonicalName());
		//loggingClassNames.add(MatrixModness.class.getCanonicalName());
		//loggingClassNames.add(RankedCC.class.getCanonicalName());
		loggingClassNames.add(NeuralNet.class.getCanonicalName());
		//loggingClassNames.add(NNHBS.class.getCanonicalName());
		loggingClassNames.add(MLSystemsManager.class.getCanonicalName());
		loggingClassNames.add(MetadataPrinter.class.getCanonicalName());
		loggingClassNames.add(Plotter.class.getCanonicalName());
		//loggingClassNames.add(MODAccuracyMeasure.class.getCanonicalName());
		//loggingClassNames.add(RelationAccuracyMeasure.class.getCanonicalName());
		//loggingClassNames.add(HMONN.class.getCanonicalName());
		//loggingClassNames.add(IndependentNN.class.getCanonicalName());
		//loggingClassNames.add(NBestAccuracy.class.getCanonicalName());
		//loggingClassNames.add(CrossLayerNN.class.getCanonicalName());
		//loggingClassNames.add(CombineIO.class.getCanonicalName());
		//loggingClassNames.add(BestIndependent.class.getCanonicalName());
		//loggingClassNames.add(FeatureWrapper.class.getCanonicalName());
		//loggingClassNames.add(MaxWeightEnsemble.class.getCanonicalName());
		//loggingClassNames.add(FullConditional.class.getCanonicalName());
		//loggingClassNames.add(WOVEnsemble.class.getCanonicalName());
		
		
		tabs = new String();
		lastChar = '\n';
	}

	private Logger()
	{
	}
	
	public static void addLoggingClassName(Class<? extends Object> c)
	{
		loggingClassNames.add(c.getCanonicalName());
	}
	
	public static synchronized void format(String format, Object... args)
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			String str = String.format(format, args);
			str = addTabs(str);
			out.print(str);
			lastChar = str.charAt(str.length()-1);
		}
	}
	
	public static synchronized void print(String str)
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			
			str = addTabs(str);
			out.print(str);
			if (!str.isEmpty())
				lastChar = str.charAt(str.length()-1);
		}
	}

	public static synchronized void println(String str)
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			str = addTabs(str);
			out.println(str);
			lastChar = '\n';
		}
	}
	
	public static synchronized void println(Object obj)
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			String str = obj.toString();
			str = addTabs(str);
			out.println(str);
			lastChar = '\n';
		}
	}

	
	public static synchronized void println()
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			out.println();
			lastChar = '\n';
		}
	}
		
	public static synchronized void printArray(String title, double[] vect)
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			String str = Helper.printArray(title, vect);
			str = addTabs(str);
			out.print(str);
			lastChar = str.charAt(str.length()-1);
		}
	}

	public static synchronized void printArray2D(String title, double[][] vect)
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			String str = Helper.arrayToString2D(title, vect);
			str = addTabs(str);
			out.print(str);
			lastChar = str.charAt(str.length()-1);
		}
	}

	private static String addTabs(String str)
	{
		// Don't add tabs at the end of str if it ends in "\n".
		if (str.length() > 0)
		{
			if (str.charAt(str.length() - 1) == '\n')
			{
				str = str.replace("\n", "\n" + tabs);
				str = str.substring(0, str.length() - tabs.length());
			}
			else
			{
				str = str.replace("\n", tabs + "\n");
			}
		}

		// Only add initial tabs if the last thing printed was a "\n".
		if (lastChar == '\n')
		{
			str = tabs + str;
		}
		return str;

	}

	
	public static synchronized void indent()
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			tabs += "\t";
		}
	}
	
	public static synchronized void unindent()
	{
		StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
		StackTraceElement e = stacktrace[2];
		if (loggingClassNames.contains(e.getClassName()))
		{
			if (tabs.length() == 0)
				throw new IllegalStateException();
			tabs = tabs.substring(0, tabs.length() - 1);
		}
	}

	

}
