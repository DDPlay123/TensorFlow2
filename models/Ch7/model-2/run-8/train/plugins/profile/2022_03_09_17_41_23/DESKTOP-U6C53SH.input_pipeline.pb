	?52;QA@?52;QA@!?52;QA@	z??1g??z??1g??!z??1g??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?52;QA@Н`?u?/@1??;???1@A\ A?c̝?I?H?"i7??Y?k?????*	effff?@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???&S??!A????xP@)?q?????1????mI@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??e??a???!?Y?+?;@)??MbX??1s???.@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?,Ԛ????!1?9?a.@),Ԛ????11?9?a.@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?	??g????!?@v?:H(@)	??g????1?@v?:H(@:Preprocessing2F
Iterator::ModelHP?sע?!?71??@)?|a2U??1?-Q ?@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch{?G?zt?!崊????){?G?zt?1崊????:Preprocessing2P
Iterator::Model::Prefetchn??t?!26w+x??)n??t?126w+x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9z??1g??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Н`?u?/@Н`?u?/@!Н`?u?/@      ??!       "	??;???1@??;???1@!??;???1@*      ??!       2	\ A?c̝?\ A?c̝?!\ A?c̝?:	?H?"i7???H?"i7??!?H?"i7??B      ??!       J	?k??????k?????!?k?????R      ??!       Z	?k??????k?????!?k?????JGPUYz??1g??b 