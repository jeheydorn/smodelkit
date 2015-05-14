package smodelkit;

/**
 * Removes instances which have unknown outputs. Instances with unknown inputs may be retained
 * (the algorithm doesn't care if inputs are unknown).
 * @author joseph
 *
 */
public class UnknownRemover
{
	public static Matrix removeRowsWithUnknownOutputs(Matrix data)
	{
		Matrix result = new Matrix();
		result.copyMetadata(data);
		
		for (int r = 0; r < data.rows(); r++)
		{
			if (!containsUnknowns(data.row(r), data.getNumLabelColumns()))
			{
				result.addRow(data.row(r));
			}
		}
		return result;
	}
	
	public static boolean containsUnknowns(Vector row, int outputCols)
	{
		for (int c = row.size() - outputCols; c < row.size(); c++)
		{
			if (Vector.isUnknown(row.get(c)))
			{
				return true;
			}
		}
		return false;
	}
}
