?	?????A@?????A@!?????A@	DRi?)??DRi?)??!DRi?)??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?????A@$a?N"?0@1J
,?)2@Aj?t???I??$??\??Y-??\n0??*	43333??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle??3??7??!?w???O@)?ڊ?e???1ㄊ?hF@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch?S??:??!_??l??2@)S??:??1_??l??2@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache??8??m4??!9?񑜅?@)??e?c]??1???4Ax2@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?]?Fx??!=????*@)]?Fx??1=????*@:Preprocessing2F
Iterator::Model6?;Nё??!h???d?@)??@??ǘ?1??a?"@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch;?O??nr?!?0??);?O??nr?1?0??:Preprocessing2P
Iterator::Model::Prefetch???_vOn?!B?TO????)???_vOn?1B?TO????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9CRi?)??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	$a?N"?0@$a?N"?0@!$a?N"?0@      ??!       "	J
,?)2@J
,?)2@!J
,?)2@*      ??!       2	j?t???j?t???!j?t???:	??$??\????$??\??!??$??\??B      ??!       J	-??\n0??-??\n0??!-??\n0??R      ??!       Z	-??\n0??-??\n0??!-??\n0??JGPUYCRi?)??b ?"b
9gradient_tape/model/conv2d_152/Conv2D/Conv2DBackpropInputConv2DBackpropInput¬[??@??!¬[??@??"-
IteratorGetNext/_1_Send??UAOܻ?!??X????"6
model/re_lu_182/Relu_FusedConv2D?|I?p??!?~?vc??"d
:gradient_tape/model/conv2d_152/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter׹?nh??!???p?=??"d
:gradient_tape/model/conv2d_153/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9?-??ֲ?!&F???"6
model/re_lu_183/Relu_FusedConv2D?^f?????!bUt$????"b
9gradient_tape/model/conv2d_153/Conv2D/Conv2DBackpropInputConv2DBackpropInput3?o?r??!H
V???"d
:gradient_tape/model/conv2d_151/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??n?[???!B??????"b
9gradient_tape/model/conv2d_151/Conv2D/Conv2DBackpropInputConv2DBackpropInput*?G}u??!5u?}	=??"6
model/re_lu_181/Relu_FusedConv2DM????ҥ?!ڰ`[6???Q      Y@Y???z?V@a;"_)) @qs??e?N5@y????d??"?	
both?Your program is POTENTIALLY input-bound because 46.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?21.3083% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 