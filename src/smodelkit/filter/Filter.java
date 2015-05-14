package smodelkit.filter;

import java.io.Serializable;
import java.util.Random;

import smodelkit.Matrix;
import smodelkit.Vector;

// Abstract class for transforming the dataset.
public abstract class Filter implements Serializable
{
	private static final long serialVersionUID = 1L;
	private Filter innerFilter;

	// Trains the filter.
	public abstract void initializeInternal(Matrix inputs, Matrix labels);

	// Filters a single input vector.
	protected abstract Vector filterInputInternal(Vector before);

	// Unfilters the predicted label(s).
	protected abstract Vector unfilterLabelInternal(Vector before);
	
	protected abstract Vector filterLabelInternal(Vector before);
	
	protected Random rand;

	/**
	 * Filters a matrix of inputs. If a sub-class needs to change the
	 * meta-data of the result, it must override this method.
	 */
	protected Matrix filterInputsInternal(Matrix inputs)
	{
		Matrix pOut = new Matrix();
		pOut.copyMetadata(inputs);
		for(int i = 0; i < inputs.rows(); i++)
		{
			Vector row = filterInputInternal(inputs.row(i));
			pOut.addRow(row);
		}
		return pOut;
	}

	/**
	 * Filters a matrix of labels. If a sub-class needs to change the
	 * meta-data of the result, it must override this method.
	 */
	protected Matrix filterLabelsInternal(Matrix labels)
	{
		Matrix pOut = new Matrix();
		pOut.copyMetadata(labels);
		for(int i = 0; i < labels.rows(); i++)
		{
			Vector row = filterLabelInternal(labels.row(i));
			pOut.addRow(row);
		}
		return pOut;

	}

	public Filter()
	{
		
	}
	
	public final void initialize(Matrix inputs, Matrix labels)
	{
		initializeInternal(inputs, labels);
		if (innerFilter != null)
		{
			Matrix pTrainInputs = filterInputsInternal(inputs);
			Matrix pTrainLabels = filterLabelsInternal(labels);
			innerFilter.initialize(pTrainInputs, pTrainLabels);
		}
	}

	public final Vector filterInput(Vector before)
	{
		if (innerFilter != null)
		{
			Vector d = filterInputInternal(before);
			Vector result = innerFilter.filterInput(d);
			return result;
		}
		else
		{
			return filterInputInternal(before);
		}		
	}

	public final Matrix filterAllInputs(Matrix inputs)
	{
		if (innerFilter != null)
		{
			Matrix m1 = filterInputsInternal(inputs);
			Matrix result = innerFilter.filterAllInputs(m1);
			return result;
		}
		else
		{
			return filterInputsInternal(inputs);
		}
	}

	public final Matrix filterAllLabels(Matrix labels)
	{
		if (innerFilter != null)
		{
			Matrix m1 = filterLabelsInternal(labels);
			Matrix result = innerFilter.filterAllLabels(m1);
			return result;
		}
		else
		{
			return filterLabelsInternal(labels);
		}
	}

	// Unfilters the predicted label(s). This also calls unfilterLabels on the
	// internalFilter.
	public final Vector unfilterLabel(Vector before)
	{
		if (innerFilter != null)
		{
			// First have the innerFilter unfilter the label, then do it myself.
			Vector innerResult = innerFilter.unfilterLabel(before);
			return unfilterLabelInternal(innerResult);
		}
		else
		{
			return unfilterLabelInternal(before);
		}
	}
	
	/**
	 * Determines if this filter is of the given class, or if any sub-filter of this filter is of the
	 * given class.
	 * @param filterCls the filter class to search for.
	 */
	public <T extends Filter> boolean includes(Class<T> filterCls)
	{
		return findFilter(filterCls) != null;
	}
	
	/**
	 * Determines if this filter is of the given class, or if any sub-filter of this filter is of the
	 * given class.
	 * @param filterCls the filter class to search for.
	 * @return Either this filter or a sub-filter of it which is of the specified type. If no filter of
	 * that type is found, then null is returned.
	 */
	@SuppressWarnings("unchecked")
	public <T extends Filter> T findFilter(Class<T> filterCls)
	{
		if (filterCls.isInstance(this))
		{
			return (T)this;
		}
		else
		{
			if (innerFilter == null)
				return null;
			return innerFilter.findFilter(filterCls);
		}
	}
	
	/**
	 * Configure this filter from arguments that were with it's name in a settings file.
	 * @param args
	 */
	public abstract void configure(String[] args);
	
	public void setRandom(Random r)
	{
		this.rand = r;
	}
	
	public void setInnerFilter(Filter inner)
	{
		this.innerFilter = inner;
	}
	
}
