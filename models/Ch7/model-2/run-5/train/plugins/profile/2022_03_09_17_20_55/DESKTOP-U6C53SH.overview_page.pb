?	??8?#OD@??8?#OD@!??8?#OD@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??8?#OD@????@a3@1??,z/4@A??????I??ދ/Z??*	gffff??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???K7?A??!??=?$4O@)?K7?A`??1?h?`ZF@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch???Q????!??s??1@)??Q????1??s??1@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache???<,Ԛ??!?j?;\=@)??+e???1Ơ?RY0@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?jM??S??!֓5?&*@)jM??S??1֓5?&*@:Preprocessing2F
Iterator::ModeltF??_??!??D)Z@)??g??s??1?H@???@:Preprocessing2P
Iterator::Model::Prefetch?+e?Xw?!?#0???)?+e?Xw?1?#0???:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch	?^)?p?!????????)	?^)?p?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????@a3@????@a3@!????@a3@      ??!       "	??,z/4@??,z/4@!??,z/4@*      ??!       2	????????????!??????:	??ދ/Z????ދ/Z??!??ދ/Z??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"5
model/re_lu_63/Relu_FusedConv2D?`Rߡ???!?`Rߡ???"-
IteratorGetNext/_1_Send?}??<???!???! ??"a
8gradient_tape/model/conv2d_52/Conv2D/Conv2DBackpropInputConv2DBackpropInput?fN????!u?y5@???"5
model/re_lu_62/Relu_FusedConv2D?>??c???!?--?{??"c
9gradient_tape/model/conv2d_50/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??%W???!D?n?????"c
9gradient_tape/model/conv2d_52/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???<-??!????????"c
9gradient_tape/model/conv2d_53/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??D??!(
?M???"a
8gradient_tape/model/conv2d_53/Conv2D/Conv2DBackpropInputConv2DBackpropInput??G(?ۭ?!??????"c
9gradient_tape/model/conv2d_51/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!s????"a
8gradient_tape/model/conv2d_51/Conv2D/Conv2DBackpropInputConv2DBackpropInput?-U?s??!?LVؽ???Q      Y@Y??????U@a!"""""*@q?.??F@y5'mYP???"?	
both?Your program is POTENTIALLY input-bound because 47.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?45.333% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 