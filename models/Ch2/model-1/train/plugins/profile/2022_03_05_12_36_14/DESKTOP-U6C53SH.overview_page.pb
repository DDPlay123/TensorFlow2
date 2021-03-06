?	֋??hW??֋??hW??!֋??hW??	?cV?w???cV?w??!?cV?w??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6֋??hW????R????1C˺,D??A?y?'L??I??5>????Y}??F??*	?????LO@2F
Iterator::Model?(??0??!nJ?-?C@)??A?f??1??h?{?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???_vO??!?$ke?7@)-C??6??1]?:?r4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatey?&1???!6??W=]6@)M??St$??1??C2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ݓ??Z??!????1>@)n??t?1???UO@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??~j?t??!??2??YN@)"??u??q?1?-??y@:Preprocessing2U
Iterator::Model::ParallelMapV2???_vOn?!?$ke?@)???_vOn?1?$ke?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vOf?!?.??@@)??_vOf?1?.??@@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!?}??!?	@)????Mb`?1?}??!?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?68.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?cV?w??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??R??????R????!??R????      ??!       "	C˺,D??C˺,D??!C˺,D??*      ??!       2	?y?'L???y?'L??!?y?'L??:	??5>??????5>????!??5>????B      ??!       J	}??F??}??F??!}??F??R      ??!       Z	}??F??}??F??!}??F??JGPUY?cV?w??b ?"#
Adam/addAddV2uj֫???!uj֫???"A
%gradient_tape/sequential/dense/MatMulMatMul?=Wb?t??!T6????"E
)gradient_tape/sequential/dense_1/MatMul_1MatMule??^?)??!`i?%Sg??"C
'gradient_tape/sequential/dense_1/MatMulMatMul??????!?b	?
??"5
sequential/dense_1/MatMulMatMul??4t??!^ ??????"3
sequential/dense/MatMulMatMul?A{??$??!?h?\4??"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchuj֫???!?
????"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamO???ؖ?!%?|????"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam&?R[?ޕ?!?/Q?~???"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdamQ??٬??!??rL???Q      Y@Y?18??5@a??18?S@q5?bs?G@y??4?e???"?
both?Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?68.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?47.0895% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 