settingsNames=(neuralnet_sigmoid_contrast.json neuralnet_sigmoid.json neuralnet_softsign_contrast.json neuralnet_softsign.json)

for settings in ${settingsNames[@]}
do
	java -Xmx6g -jar manager.jar -L neuralnet model_settings/$settings -A Datasets/mcc/CIFAR-10_train.arff -E static Datasets/mcc/CIFAR-10_test.arff > $settings.out 2> $settings.err &
done
