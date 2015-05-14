package smodelkit.util;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * This class stores data for plots, then generates those plots by calling
 * generateAllPlots(). It uses MatplotLib to draw plots.
 * @author joseph
 *
 */
public class Plotter
{
	private static Map<String, Plot> plots;
	private enum PlotType
	{
		line,
		histogram
	};
	
	static
	{
		plots = new TreeMap<>();
	}
	
	private Plotter()
	{
		
	}
		
	/**
	 * If plotName doesn't exist, this creates a plot for it. The datum given is added to the list
	 * of data for that plot. The datum must be the same size as all others for that plot.
	 * 
	 * @param datum A data point to plot. If this has length > 1, then multiple lines
	 * will be drawn on the same plot.
	 */
	public static void addDatumForLinePlot(String plotName, double[] datum, String xLabel, String yLabel)
	{
		Plot plot = plots.get(plotName);
		if (plot == null)
		{
			plot = new Plot(xLabel, yLabel, PlotType.line);
			plots.put(plotName, plot);
		}
		
		if (plot.data.size() > 0 && datum.length != plot.data.get(0).length)
			throw new IllegalArgumentException("All datums must be of the same dimensions to be plotted.");
		plot.data.add(Arrays.copyOf(datum, datum.length));
	}

	/**
	 * If plotName doesn't exist, this creates a plot for it. The datum given is added to the list
	 * of data for that plot.
	 * 
	 * @param datum A data point to add to the histogram to plot.
	 * @param binCount The number of bins to plot with. This must be the same every time this
	 * function is called with the same plotName.
	 */
	public static void addDatumForHistogramPlot(String plotName, double datum, String xLabel, String yLabel, int binCount)
	{
		Plot plot = plots.get(plotName);
		if (plot == null)
		{
			plot = new Plot(xLabel, yLabel, PlotType.histogram);
			plot.binCount = binCount;
			plots.put(plotName, plot);
		}
		else if (plot.data.size() > 0 && plot.binCount != binCount)
		{
			throw new IllegalArgumentException("You must give the same binCount whenever the same plotName is used.");
		}
		
		plot.data.add(new double[] {datum});
	}

	public static void generateAllPlots()
	{
		for (String plotName : plots.keySet())
		{
			Logger.println("Creating plot " + plotName);
			Plot plot = plots.get(plotName);
			
			String filename = plotName + ".csv";
			try (PrintWriter p = new PrintWriter(filename))
			{
				p.println(plot.xLabel + "," + plot.yLabel);
				
				for (double[] datum : plot.data)
				{
					for (int d : new Range(datum.length))
					{
						p.print(datum[d]);
						if (d < datum.length - 1)
							p.print(",");
					}
					p.println();
				}
			} 
			catch (FileNotFoundException e)
			{
				System.out.println("Unable to generate plot due to expeption: ");
				e.printStackTrace();
			}
			
			if (plot.type.equals(PlotType.line))
			{
				Helper.executeCommand("python plotLine.py " + filename);				
			}
			else if (plot.type.equals(PlotType.histogram))
			{
				Helper.executeCommand("python plotHistogram.py " + filename);							
			}
			else
			{
				throw new IllegalArgumentException("Unrecognized plot type: " + plot.type);
			}
		}
	}
	
	private static class Plot
	{
		List<double[]> data;
		String xLabel;
		String yLabel;
		PlotType type;
		int binCount;

		public Plot(String xLabel, String yLabel, PlotType type)
		{
			this.data = new ArrayList<>();
			this.xLabel = xLabel;
			this.yLabel = yLabel;
			this.type = type;
		}	
	}
	
}
