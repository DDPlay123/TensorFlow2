	Pr?Md?5@Pr?Md?5@!Pr?Md?5@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Pr?Md?5@?%?<??1??0B`3@AP ?Ȓ9??IV??f???*	33333Cd@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle?D?l?????!??}HU@)D?l?????1??}HU@:Preprocessing2F
Iterator::Model???{????!??Im$@)A??ǘ???1oL???c@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?q?????!???,T?@)?q?????1???,T?@:Preprocessing2P
Iterator::Model::Prefetch??_?Lu?!?"?ũ	@)??_?Lu?1?"?ũ	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?%?<???%?<??!?%?<??      ??!       "	??0B`3@??0B`3@!??0B`3@*      ??!       2	P ?Ȓ9??P ?Ȓ9??!P ?Ȓ9??:	V??f???V??f???!V??f???B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 