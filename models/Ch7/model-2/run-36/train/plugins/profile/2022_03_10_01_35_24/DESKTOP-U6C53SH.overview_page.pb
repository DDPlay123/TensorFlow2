?	Y|E?L@Y|E?L@!Y|E?L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Y|E?L@?C?X?@@1P???bl7@A?{??Pk??I'?Ҩ ??*	23333??@2g
/Iterator::Model::Prefetch::MapAndBatch::Shuffle???ݓ????!Z??
jQ@)?Pk?w???1???'?G@:Preprocessing2q
9Iterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch???????!y???YY4@)??????1y???YY4@:Preprocessing2~
FIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCache?%u???!?>?n<@)鷯猸?1??Qē/@:Preprocessing2?
JIterator::Model::Prefetch::MapAndBatch::Shuffle::Prefetch::MemoryCacheImpl?a2U0*???!'ˎ??I)@)a2U0*???1'ˎ??I)@:Preprocessing2F
Iterator::Model???{????!"??/?w@)a??+e??1ލ???T @:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch????Mbp?!?S?(???)????Mbp?1?S?(???:Preprocessing2P
Iterator::Model::Prefetch?q????o?!?t????)?q????o?1?t????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?C?X?@@?C?X?@@!?C?X?@@      ??!       "	P???bl7@P???bl7@!P???bl7@*      ??!       2	?{??Pk???{??Pk??!?{??Pk??:	'?Ҩ ??'?Ҩ ??!'?Ҩ ??B      ??!       J      ??!       R      ??!       Z      ??!       JGPUb ?"b
9gradient_tape/model/conv2d_207/Conv2D/Conv2DBackpropInputConv2DBackpropInput??O????!??O????"-
IteratorGetNext/_2_RecvLtW2u??!??qkC??"d
:gradient_tape/model/conv2d_207/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????Qm??!??q?????"3
model/conv2d_207/Conv2DConv2D"Q?!(!ú]???"d
:gradient_tape/model/conv2d_208/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?um????!??qZC??"b
9gradient_tape/model/conv2d_208/Conv2D/Conv2DBackpropInputConv2DBackpropInput?"?????!rv?????"3
model/conv2d_208/Conv2DConv2D?? ?)???!H??Ok??"d
:gradient_tape/model/conv2d_206/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??1?ܤ?!??	t??"j
@gradient_tape/model/batch_normalization_102/FusedBatchNormGradV3FusedBatchNormGradV3??xC?ã?!DA?????"b
9gradient_tape/model/conv2d_206/Conv2D/Conv2DBackpropInputConv2DBackpropInputd֚??h??!????=v??Q      Y@Y??ެ'?U@ay??®)@q?jaj?>@y/?]	hÆ?"?	
both?Your program is POTENTIALLY input-bound because 57.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb?30.4988% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 