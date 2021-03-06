?	?7>[?4@?7>[?4@!?7>[?4@	??KS?????KS???!??KS???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?7>[?4@E?>?/3@1*?:]???A?ׁsF???Ij?Z_$4??YCV????*	gffff&M@2F
Iterator::ModeljM????!?r?c{a@@)?!??u???1nS?ul/8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!?D|s?T=@)y?&1???1?dK?8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?? ?rh??!V?H?(=@)???S㥋?1Ѽv?'7@:Preprocessing2U
Iterator::Model::ParallelMapV2{?G?zt?!?#Y?'!@){?G?zt?1?#Y?'!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipn????!?F)NB?P@)y?&1?l?1?dK?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!?dK?@)y?&1?l?1?dK?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ei?!~???D@)a??+ei?1~???D@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapΈ?????!
=c?h??@)-C??6Z?1?7?K??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??KS???>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	E?>?/3@E?>?/3@!E?>?/3@      ??!       "	*?:]???*?:]???!*?:]???*      ??!       2	?ׁsF????ׁsF???!?ׁsF???:	j?Z_$4??j?Z_$4??!j?Z_$4??B      ??!       J	CV????CV????!CV????R      ??!       Z	CV????CV????!CV????JGPUY??KS???b ?"G
+gradient_tape/sequential_1/dense_4/MatMul_1MatMul
8??M??!
8??M??"E
)gradient_tape/sequential_1/dense_3/MatMulMatMul???7X6??!L?f?A??"#
Adam/addAddV2he?I??! !":s??"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchҕy?#??!?Sѷ????"W
6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGradBiasAddGrad?O?uR???!?]?????"7
sequential_1/dense_3/MatMulMatMul?f?A0??!??????"W
6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGradBiasAddGrad0?:r/??!??v???"E
)gradient_tape/sequential_1/dense_4/MatMulMatMul$?q?????!??ǩje??"7
sequential_1/dense_4/MatMulMatMul$?q?????!??ՙ???"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamWN?NRѕ?!fa???-??Q      Y@Y?18??5@a??18?S@qL?????O@y?!?M?f??"?
both?Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?63.623% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 