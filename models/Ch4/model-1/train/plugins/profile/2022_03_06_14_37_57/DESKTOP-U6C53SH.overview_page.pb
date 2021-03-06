?	?1˞?@?1˞?@!?1˞?@	?<'蜨<@?<'蜨<@!?<'蜨<@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?1˞?@???c?R??1w.???v??A???ω?I?*n?b???Y????#???*	     0e@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??z?G???!%PR?U@)?z?G???1%PR?U@:Preprocessing2F
Iterator::Modeln????!?r?x '@)????Mb??1?.?	?"@:Preprocessing2P
Iterator::Model::Prefetch??H?}m?!??	?? @)??H?}m?1??	?? @:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatcha2U0*?c?!Vzja????)a2U0*?c?1Vzja????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 28.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?30.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t13.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?<'蜨<@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???c?R?????c?R??!???c?R??      ??!       "	w.???v??w.???v??!w.???v??*      ??!       2	???ω????ω?!???ω?:	?*n?b????*n?b???!?*n?b???B      ??!       J	????#???????#???!????#???R      ??!       Z	????#???????#???!????#???JGPUY?<'蜨<@b ?"-
IteratorGetNext/_1_Send??j?-???!??j?-???"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam؎?ָ[??!?";????"3
sequential/dense/MatMulMatMul!@NC?u??!?꤮b???"K
$Adam/Adam/update_6/ResourceApplyAdamResourceApplyAdam??y??8??!????"A
%gradient_tape/sequential/dense/MatMulMatMulٿ?#5??!~???????"C
'gradient_tape/sequential/dense_3/MatMulMatMul?n??r??!?=??r???"5
sequential/dense_3/MatMulMatMul? ???R??!???5??"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam???_??!w?*???"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdam?nOA???!?8?\??"E
)gradient_tape/sequential/dense_3/MatMul_1MatMul%PP?≑?!m???k???Q      Y@Y????2?P@a??\??@@q???'?%.@y?w?"S???"?
host?Your program is HIGHLY input-bound because 28.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?30.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t13.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?15.0737% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 