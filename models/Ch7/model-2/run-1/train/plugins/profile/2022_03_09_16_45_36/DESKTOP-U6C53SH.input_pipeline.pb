	?y?S?BC@?y?S?BC@!?y?S?BC@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?y?S?BC@D?|?~1@1b????3@AA??ǘ???I??yr??*	    ??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?<?R?!???!&??yP@)?Zd;???1G?tj?I@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?yX?5?;??!?Q`ҩ<@)?(\?????15r??0@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?A??ǘ???!߈?N-@)A??ǘ???1߈?N-@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?????ׁ??!Y?&)@)????ׁ??1Y?&)@:Preprocessing2F
Iterator::Model?#??????!?X?@)A??ǘ???1߈?N@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?+e?Xw?!??~?X??)?+e?Xw?1??~?X??:Preprocessing2P
Iterator::Model::Prefetch??ZӼ?t?!?I??A???)??ZӼ?t?1?I??A???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	D?|?~1@D?|?~1@!D?|?~1@      ??!       "	b????3@b????3@!b????3@*      ??!       2	A??ǘ???A??ǘ???!A??ǘ???:	??yr????yr??!??yr??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 