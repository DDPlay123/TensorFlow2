	?HM???B@?HM???B@!?HM???B@	? ????? ????!? ????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?HM???B@1???61@1g????3@A?X?? ??I:?S?????Y?>:u峌?*	??????@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??ZӼ???!G x??J@)??4?8E??1˂?RthD@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?	?c???!?Y?-~D@)}??b???1???u?#7@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??c]?F??!?,??F?0@)?c]?F??1?,??F?0@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch??z6?>??!???/??(@)?z6?>??1???/??(@:Preprocessing2F
Iterator::ModelM??St$??!??4?@){?G?z??1???d?@:Preprocessing2P
Iterator::Model::Prefetch??_?Lu?!????|???)??_?Lu?1????|???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatcha2U0*?s?!?d??K???)a2U0*?s?1?d??K???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9? ????>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	1???61@1???61@!1???61@      ??!       "	g????3@g????3@!g????3@*      ??!       2	?X?? ???X?? ??!?X?? ??:	:?S?????:?S?????!:?S?????B      ??!       J	?>:u峌??>:u峌?!?>:u峌?R      ??!       Z	?>:u峌??>:u峌?!?>:u峌?JGPUY? ????b 