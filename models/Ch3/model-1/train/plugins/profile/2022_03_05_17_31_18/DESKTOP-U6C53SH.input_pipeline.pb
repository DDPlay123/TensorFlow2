	-@?j@-@?j@!-@?j@	?L?X@?L?X@!?L?X@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6-@?j@L?Qԙ??1OWw,?I??A???????Ica???^??Y?K6l???*	     ?P@2F
Iterator::ModelǺ????!6?d?M?G@)9??v????1??&?l?C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-???!???>?5@)?HP???1|??|2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?(??0??!??.???2@)U???N@??1|??|,@:Preprocessing2U
Iterator::Model::ParallelMapV2?g??s?u?!>??? @)?g??s?u?1>??? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??W?2ġ?!?&?l?IJ@)"??u??q?1>???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???H??!]t?E8@)??H?}m?1?E]t?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????g?!?M6?d?@)?????g?1?M6?d?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!??>??@)HP?s?b?1??>??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?60.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t17.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?L?X@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L?Qԙ??L?Qԙ??!L?Qԙ??      ??!       "	OWw,?I??OWw,?I??!OWw,?I??*      ??!       2	??????????????!???????:	ca???^??ca???^??!ca???^??B      ??!       J	?K6l????K6l???!?K6l???R      ??!       Z	?K6l????K6l???!?K6l???JGPUY?L?X@b 