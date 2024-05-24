#include "cnn_mnist_model.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#define TF_NUM_OPS 4 
#define TF_NUM_INPUTS 64  // 8x8 flattened input
#define TF_NUM_OUTPUTS 10  // 10 classes (digits 0-9)
#define TF_OP_CONV2D
#define TF_OP_FULLYCONNECTED
#define TF_OP_SOFTMAX
#define TF_OP_RESHAPE

#define ARENA_SIZE 20000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

void setup() {
    Serial.begin(115200);
    delay(3000);
    Serial.println("__TENSORFLOW MNIST__");

    // configure input/output
    tf.setNumInputs(64);  // 8x8 flattened input
    tf.setNumOutputs(10);  // 10 classes (digits 0-9)

    // add required ops (order does not need to match model architecture)
    tf.resolver.AddConv2D();
    tf.resolver.AddFullyConnected();
    tf.resolver.AddSoftmax();
    tf.resolver.AddReshape(); // Add the Reshape operation if used

    while (!tf.begin(cnn_mnist_model_tflite).isOk()) 
        Serial.println(tf.exception.toString());

    // Print model info
    Serial.println("Model initialized.");
}

void loop() {
    // MNIST input data for digit "8"
    // Digit 0
    float x_test0[64] = {
      0.000000f, 0.000000f, 0.312500f, 0.812500f, 0.562500f, 0.062500f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.812500f, 0.937500f, 0.625000f, 0.937500f, 0.312500f, 0.000000f, 
      0.000000f, 0.187500f, 0.937500f, 0.125000f, 0.000000f, 0.687500f, 0.500000f, 0.000000f, 
      0.000000f, 0.250000f, 0.750000f, 0.000000f, 0.000000f, 0.500000f, 0.500000f, 0.000000f, 
      0.000000f, 0.312500f, 0.500000f, 0.000000f, 0.000000f, 0.562500f, 0.500000f, 0.000000f, 
      0.000000f, 0.250000f, 0.687500f, 0.000000f, 0.062500f, 0.750000f, 0.437500f, 0.000000f, 
      0.000000f, 0.125000f, 0.875000f, 0.312500f, 0.625000f, 0.750000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.375000f, 0.812500f, 0.625000f, 0.000000f, 0.000000f, 0.000000f, 
      };

    // Digit 1
    float x_test1[64] = {
      0.000000f, 0.000000f, 0.000000f, 0.750000f, 0.812500f, 0.312500f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.687500f, 1.000000f, 0.562500f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.187500f, 0.937500f, 1.000000f, 0.375000f, 0.000000f, 0.000000f, 
      0.000000f, 0.437500f, 0.937500f, 1.000000f, 1.000000f, 0.125000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.062500f, 1.000000f, 1.000000f, 0.187500f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.062500f, 1.000000f, 1.000000f, 0.375000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.062500f, 1.000000f, 1.000000f, 0.375000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.687500f, 1.000000f, 0.625000f, 0.000000f, 0.000000f, 
      };

    // Digit 2
    float x_test2[64] = {
      0.000000f, 0.000000f, 0.000000f, 0.250000f, 0.937500f, 0.750000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.187500f, 1.000000f, 0.937500f, 0.875000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.500000f, 0.812500f, 0.500000f, 1.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.062500f, 0.375000f, 0.937500f, 0.687500f, 0.000000f, 0.000000f, 
      0.000000f, 0.062500f, 0.500000f, 0.812500f, 0.937500f, 0.062500f, 0.000000f, 0.000000f, 
      0.000000f, 0.562500f, 1.000000f, 1.000000f, 0.312500f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.187500f, 0.812500f, 1.000000f, 1.000000f, 0.687500f, 0.312500f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.187500f, 0.687500f, 1.000000f, 0.562500f, 0.000000f, 
      };

    // Digit 3
    float x_test3[64] = {
      0.000000f, 0.000000f, 0.437500f, 0.937500f, 0.812500f, 0.062500f, 0.000000f, 0.000000f, 
      0.000000f, 0.500000f, 0.812500f, 0.375000f, 0.937500f, 0.250000f, 0.000000f, 0.000000f, 
      0.000000f, 0.125000f, 0.062500f, 0.812500f, 0.812500f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.125000f, 0.937500f, 0.687500f, 0.062500f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.062500f, 0.750000f, 0.750000f, 0.062500f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.062500f, 0.625000f, 0.500000f, 0.000000f, 
      0.000000f, 0.000000f, 0.500000f, 0.250000f, 0.312500f, 0.875000f, 0.562500f, 0.000000f, 
      0.000000f, 0.000000f, 0.437500f, 0.812500f, 0.812500f, 0.562500f, 0.000000f, 0.000000f, 
      };

    // Digit 4
    float x_test4[64] = {
      0.000000f, 0.000000f, 0.000000f, 0.062500f, 0.687500f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.437500f, 0.500000f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.062500f, 0.812500f, 0.375000f, 0.125000f, 0.125000f, 0.000000f, 
      0.000000f, 0.000000f, 0.437500f, 0.937500f, 0.000000f, 0.562500f, 0.500000f, 0.000000f, 
      0.000000f, 0.312500f, 1.000000f, 0.625000f, 0.000000f, 1.000000f, 0.375000f, 0.000000f, 
      0.000000f, 0.250000f, 0.937500f, 1.000000f, 0.812500f, 1.000000f, 0.062500f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.187500f, 0.937500f, 0.625000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.125000f, 1.000000f, 0.250000f, 0.000000f, 0.000000f, 
      };

    // Digit 5
    float x_test5[64] = {
      0.000000f, 0.000000f, 0.750000f, 0.625000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.875000f, 1.000000f, 1.000000f, 0.875000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.812500f, 1.000000f, 0.937500f, 0.625000f, 0.062500f, 0.000000f, 
      0.000000f, 0.000000f, 0.687500f, 1.000000f, 1.000000f, 0.437500f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.250000f, 0.437500f, 1.000000f, 0.437500f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.250000f, 1.000000f, 0.562500f, 0.000000f, 
      0.000000f, 0.000000f, 0.312500f, 0.250000f, 0.750000f, 1.000000f, 0.250000f, 0.000000f, 
      0.000000f, 0.000000f, 0.562500f, 1.000000f, 1.000000f, 0.625000f, 0.000000f, 0.000000f, 
      };

    // Digit 6
    float x_test6[64] = {
      0.000000f, 0.000000f, 0.000000f, 0.750000f, 0.812500f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.312500f, 1.000000f, 0.500000f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.812500f, 1.000000f, 0.187500f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.875000f, 0.812500f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.937500f, 0.750000f, 0.437500f, 0.125000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.812500f, 1.000000f, 0.812500f, 1.000000f, 0.187500f, 0.000000f, 
      0.000000f, 0.000000f, 0.437500f, 1.000000f, 0.687500f, 0.937500f, 0.500000f, 0.000000f, 
      0.000000f, 0.000000f, 0.062500f, 0.562500f, 0.937500f, 0.687500f, 0.187500f, 0.000000f, 
      };

    // Digit 7
    float x_test7[64] = {
      0.000000f, 0.000000f, 0.437500f, 0.500000f, 0.812500f, 1.000000f, 0.937500f, 0.062500f, 
      0.000000f, 0.000000f, 0.437500f, 0.437500f, 0.250000f, 0.687500f, 0.750000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.500000f, 0.812500f, 0.062500f, 0.000000f, 
      0.000000f, 0.250000f, 0.500000f, 0.500000f, 0.937500f, 0.937500f, 0.375000f, 0.000000f, 
      0.000000f, 0.125000f, 0.687500f, 0.937500f, 0.937500f, 0.250000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.312500f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.562500f, 0.937500f, 0.062500f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.812500f, 0.312500f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
      };

    // Digit 8
    float x_test8[64] = {
      0.000000f, 0.000000f, 0.562500f, 0.875000f, 0.500000f, 0.062500f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.750000f, 0.875000f, 0.875000f, 0.750000f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.562500f, 0.625000f, 0.000000f, 0.937500f, 0.250000f, 0.000000f, 
      0.000000f, 0.000000f, 0.187500f, 1.000000f, 0.750000f, 0.875000f, 0.125000f, 0.000000f, 
      0.000000f, 0.000000f, 0.250000f, 1.000000f, 1.000000f, 0.125000f, 0.000000f, 0.000000f, 
      0.000000f, 0.187500f, 1.000000f, 0.500000f, 0.625000f, 0.812500f, 0.125000f, 0.000000f, 
      0.000000f, 0.062500f, 0.937500f, 0.062500f, 0.187500f, 1.000000f, 0.500000f, 0.000000f, 
      0.000000f, 0.000000f, 0.687500f, 1.000000f, 0.937500f, 0.687500f, 0.062500f, 0.000000f, 
      };

    // Digit 9
    float x_test9[64] = {
      0.000000f, 0.000000f, 0.687500f, 0.750000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 
      0.000000f, 0.125000f, 1.000000f, 1.000000f, 1.000000f, 0.812500f, 0.000000f, 0.000000f, 
      0.000000f, 0.187500f, 1.000000f, 0.750000f, 0.625000f, 0.875000f, 0.000000f, 0.000000f, 
      0.000000f, 0.062500f, 1.000000f, 0.062500f, 0.750000f, 0.937500f, 0.000000f, 0.000000f, 
      0.000000f, 0.000000f, 0.812500f, 1.000000f, 0.562500f, 0.937500f, 0.125000f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.187500f, 0.000000f, 0.562500f, 0.687500f, 0.000000f, 
      0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.562500f, 0.937500f, 0.250000f, 0.000000f, 
      0.000000f, 0.000000f, 0.562500f, 0.750000f, 0.812500f, 0.187500f, 0.000000f, 0.000000f, 
      };

    
    // classify sample input
    if (!tf.predict(x_test0).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 0, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test1).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 1, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test2).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 2, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test3).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 3, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test4).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 4, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test5).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 5, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test6).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 6, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test7).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 7, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test8).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 8, predicted digit: ");
    Serial.println(tf.classification);

    // classify sample input
    if (!tf.predict(x_test9).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    Serial.print("Expected digit: 9, predicted digit: ");
    Serial.println(tf.classification);

    // how long does it take to run a single prediction?
    Serial.print("\nIt takes ");
    Serial.print(tf.benchmark.microseconds());
    Serial.println(" us for a single prediction");

    // Calculate memory usage
    uint32_t freeHeap = ESP.getFreeHeap();
    uint32_t totalHeap = ESP.getHeapSize();
    float usedMemoryPercentage = ((float)(totalHeap - freeHeap) / totalHeap) * 100;

    Serial.print("\nMemory usage: ");
    Serial.print(usedMemoryPercentage);
    Serial.println("%\n");




    delay(1000);
}
