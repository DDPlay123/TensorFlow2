	??)r?6@??)r?6@!??)r?6@	0??%???0??%???!0??%???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??)r?6@ȔA?45@1?ܵ?|???A?]K?=??IŮ???d??Y?x???*	?????LF@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate%u???!?[P??z@@) ?o_Ή?1xI@<@:Preprocessing2F
Iterator::Model????Mb??!?ݡ??A@)?~j?t???1?W?r??:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ZӼ???!??a??6@)"??u????1C???RH3@:Preprocessing2U
Iterator::Model::ParallelMapV2????Mbp?!?ݡ??!@)????Mbp?1?ݡ??!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???QI??!@8/	P@)Ǻ???f?1????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?J?4a?!??[P??@)?J?4a?1??[P??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6Z?!?????@)-C??6Z?1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?q??????!?k?J!}A@)??H?}M?1&?D?$ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9/??%???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ȔA?45@ȔA?45@!ȔA?45@      ??!       "	?ܵ?|????ܵ?|???!?ܵ?|???*      ??!       2	?]K?=???]K?=??!?]K?=??:	Ů???d??Ů???d??!Ů???d??B      ??!       J	?x????x???!?x???R      ??!       Z	?x????x???!?x???JGPUY/??%???b 