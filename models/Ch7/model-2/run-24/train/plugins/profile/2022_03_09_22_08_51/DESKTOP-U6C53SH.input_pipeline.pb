	?ѩ+?A@?ѩ+?A@!?ѩ+?A@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?ѩ+?A@,d???0@1uʣ2@Au????I*Ŏơ>??*	     ??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??\?C????!??A.?P@)!?rh????1??mG@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?ı.n???!B??Fa3@)ı.n???1B??Fa3@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?Ӽ????!.?!?&>@)+??η?1?c??y0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl??N@aó?!?`?Z+@)?N@aó?1?`?Z+@:Preprocessing2F
Iterator::Model??ǘ????!????W?@)M??St$??17S?à @:Preprocessing2P
Iterator::Model::Prefetchn??t?!U!e?????)n??t?1U!e?????:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?J?4q?!?2????)?J?4q?1?2????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,d???0@,d???0@!,d???0@      ??!       "	uʣ2@uʣ2@!uʣ2@*      ??!       2	u????u????!u????:	*Ŏơ>??*Ŏơ>??!*Ŏơ>??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb 