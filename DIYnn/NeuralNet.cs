using Freestylecoding.Math.Linear;

namespace DIYnn {
	/// <summary>Creates a simple Neural Network</summary>
	/// <typeparam name="T">The type expected to be returned from the network</typeparam>
	internal class NeuralNet<T> {
		/// <summary>Sigmoid Function</summary>
		public static double Sigmoid( double x ) =>
			1.0 / ( 1.0 + double.Exp( -x ) );
		/// <summary>Derivative of the Sigmoid Function</summary>
		public static double SigmoidPrime( double x ) =>
			Sigmoid( x ) * ( 1.0 - Sigmoid( x ) );
		/// <summary>ReLU (Rectified Linear Unit) Function</summary>
		public static double ReLU( double x ) =>
			Math.Max( 0.0, x );
		/// <summary>Derivative of the ReLU Function</summary>
		public static double ReLUPrime( double x ) =>
			0.0 < x
				? 1.0
				: 0.0;

		private static readonly Random random = new Random();
		private static double RandomValue( int _ ) =>
			random.NextDouble();

		private Vector<double> HiddenBiases;
		private Vector<double> OutputBiases;
		private Matrix<double> InputWeights;
		private Matrix<double> HiddenWeights;

		/// <summary>The activation function for the hidden node</summary>
		public Func<double,double> HiddenActivation { get; set; }
		/// <summary>The activation function for the output node</summary>
		public Func<double,double> OutputActivation { get; set; }
		/// <summary>The derivative of the activation function for the hidden node</summary>
		/// <remarks>Used in training</remarks>
		public Func<double,double> HiddenActivationPrime { get; set; }
		/// <summary>The derivative of the activation function for the output node</summary>
		/// <remarks>Used in training</remarks>
		public Func<double,double> OutputActivationPrime { get; set; }
		/// <summary>A function that tells how to turn the output of the network into the value(s) expected by the program</summary>
		public Func<Vector<double>,T> TransformOutput { get; set; }

		/// <summary>Creates a Neural Network</summary>
		/// <param name="InputSize">The number of inputs for the network</param>
		/// <param name="HiddenSize">The number of neurons in the hidden layer</param>
		/// <param name="OutputSize">The number of neurons in the output layer</param>
		/// <param name="transformOutput">Function that takes the output layer and transforms it into <typeparamref name="T"/></param>
		/// <remarks>The values of the weights and biases are initialized to random values between 0 and 1</remarks>
		public NeuralNet( int InputSize, int HiddenSize, int OutputSize, Func<Vector<double>,T> transformOutput ) {
			HiddenBiases = new Vector<double>( HiddenSize, RandomValue );
			OutputBiases = new Vector<double>( OutputSize, RandomValue );

			InputWeights = new Matrix<double>( HiddenSize, InputSize, RandomValue );
			HiddenWeights = new Matrix<double>( OutputSize, HiddenSize, RandomValue );

			TransformOutput = transformOutput;

			HiddenActivation = Sigmoid;
			HiddenActivationPrime = SigmoidPrime;

			OutputActivation = Sigmoid;
			OutputActivationPrime = SigmoidPrime;
		}

		/// <summary>Runs a set of values through the Neural Network and returns the result</summary>
		/// <param name="Input">Vector of input values</param>
		/// <returns>The result of the calculation</returns>
		/// <remarks>This is what is considered Forward Propagation through the network</remarks>
		public T Test( Vector<double> Input ) {
			Vector<double> HiddenLayer = ( ( InputWeights * Input ) + HiddenBiases ).Apply( HiddenActivation );
			Vector<double> OutputLayer = ( ( HiddenWeights * HiddenLayer ) + OutputBiases ).Apply( OutputActivation );
			return TransformOutput( OutputLayer );
		}

		/// <summary>Adjusts the weights and biases of the network based on expected results</summary>
		/// <param name="Inputs">A tuple that contains the inputs and the expected output</param>
		/// <param name="TransformDesiredOutput">A function that takes an expected output and returns the best case values from the output layer</param>
		/// <remarks>This is what is considered Back Propagation through the network</remarks>
		public void Train( IEnumerable<(Vector<double>, T)> Inputs, Func<T,Vector<double>> TransformDesiredOutput ) {
			// Initialize the error "collectors"
			Vector<double> HiddenBiasErrors = new Vector<double>( HiddenBiases.Length, 0 );
			Vector<double> OutputBiasErrors = new Vector<double>( OutputBiases.Length, 0 ); ;

			Matrix<double> InputWeightErrors = new Matrix<double>(
				InputWeights.Rows,
				InputWeights.Columns,
				0.0
			);
			Matrix<double> HiddenWeightErrors = new Matrix<double>(
				HiddenWeights.Rows,
				HiddenWeights.Columns,
				0.0
			);

			// Calculate the error and adjustments for each input
			foreach( (Vector<double> Input, T Expected) in Inputs ) {
				// Replicates the Forward Propagation step, while retaining the intermediate values
				Vector<double> HiddenLayer = ( ( InputWeights * Input ) + HiddenBiases ).Apply( HiddenActivation );
				Vector<double> OutputLayer = ( ( HiddenWeights * HiddenLayer ) + OutputBiases ).Apply( OutputActivation );
				Vector<double> DesiredOutput = TransformDesiredOutput( Expected );

				// A Back Propagation works by using a techinque called a gradient descent on the error function
				// Basically, it creates a functions that defines how far off from the expected value the calculated error is
				// It then creates a vector that points towards the minimum value of the error

				// Since we have to do this for every value (including ones that are calculated correctly)
				// we don't want to point directly towards the lowest value.
				// Thisis because
				// 1) It is possible that we point towards a local minimum instead of the global minimum
				// 2) This value gets averaged with the results from the other training data

				// Instead, we want to slowly "nudge" the values towards the correct direction

				// Create the vector that corrects towards the minimum of the error function for the output layer
				Vector<double> GlobalError = new Vector<double>(
					Enumerable.Range( 0, OutputLayer.Length )
						.Select( i => OutputActivationPrime( OutputLayer[i] ) * ( DesiredOutput[i] - OutputLayer[i] ) )
						.ToArray()
				);

				// Adjust our output biases based on the global error
				OutputBiasErrors += GlobalError;

				// Adjust our hidden weights based on the global error
				HiddenWeightErrors += new Matrix<double>(
					Enumerable.Range( 0, GlobalError.Length )
						.Select( i => new Vector<double>( HiddenLayer * GlobalError[i] ) )
						.ToArray()
				);

				// Create the vector that corrects towards the minimum of the error function for the hidden layer
				Vector<double> HiddenError = new Vector<double>(
					Enumerable.Range( 0, HiddenLayer.Length )
						.Select( i => HiddenActivationPrime( HiddenLayer[i] ) * ( GlobalError * HiddenWeights.GetColumn( i ) ) )
						.ToArray()
				);

				// Adjust our hidden biases based on the hidden error
				HiddenBiasErrors += HiddenError;

				// Adjust our input weights based on the hidden error
				InputWeightErrors += new Matrix<double>(
					Enumerable.Range( 0, HiddenError.Length )
						.Select( i => new Vector<double>( Input * HiddenError[i] ) )
						.ToArray()
				);
			} // end of foreach loop

			// Average all our errors and adjust the weights and biases based on the result
			OutputBiases += OutputBiasErrors.Apply( d => d / Inputs.Count() );
			HiddenBiases += HiddenBiasErrors.Apply( d => d / Inputs.Count() );
			HiddenWeights += HiddenWeightErrors.Apply( d => d / Inputs.Count() );
			InputWeights += InputWeightErrors.Apply( d => d / Inputs.Count() );
		}

		/// <summary>Prints the values for the weights and biases to the console</summary>
		/// <remarks>You normally don't care about these values, but it's fun for learning purposes</remarks>
		public void PrintNetwork() {
			Console.WriteLine( $"{InputWeights:F5}" );
			Console.WriteLine();
			Console.WriteLine( $"{HiddenBiases:F5}" );
			Console.WriteLine();
			Console.WriteLine( $"{HiddenWeights:F5}" );
			Console.WriteLine();
			Console.WriteLine( $"{OutputBiases:F5}" );
		}
	}
}

/*
		public void Train( IEnumerable<(Vector<double>, T)> Inputs, Func<T,Vector<double>> TransformDesiredOutput ) {
			Vector<double> HiddenBiasErrors = new Vector<double>( HiddenBiases.Length, 0 );
			Vector<double> OutputBiasErrors = new Vector<double>( OutputBiases.Length, 0 ); ;

			Matrix<double> InputWeightErrors = new Matrix<double>(
				InputWeights.Rows,
				InputWeights.Columns,
				0.0
			);
			Matrix<double> HiddenWeightErrors = new Matrix<double>(
				HiddenWeights.Rows,
				HiddenWeights.Columns,
				0.0
			);

			foreach( (Vector<double> Input, T Expected) in Inputs ) {
				Vector<double> HiddenLayer = ( ( InputWeights * Input ) + HiddenBiases ).Apply( HiddenActivation );
				Vector<double> OutputLayer = ( ( HiddenWeights * HiddenLayer ) + OutputBiases ).Apply( OutputActivation );

				Vector<double> DesiredOutput = TransformDesiredOutput( Expected );

				Vector<double> GlobalError = new Vector<double>( new double[] {
					OutputActivationPrime( OutputLayer[0] ) * ( DesiredOutput[0] - OutputLayer[0] ),
					OutputActivationPrime( OutputLayer[1] ) * ( DesiredOutput[1] - OutputLayer[1] )
				} );

				OutputBiasErrors += GlobalError;
				HiddenWeightErrors += new Matrix<double>( new[] {
					HiddenLayer * GlobalError[0],
					HiddenLayer * GlobalError[1]
				} );

				Vector<double> HiddenError = new Vector<double>( new double[] {
					HiddenActivationPrime( HiddenLayer[0] ) * ( GlobalError * HiddenWeights.GetColumn( 0 ) ),
					HiddenActivationPrime( HiddenLayer[1] ) * ( GlobalError * HiddenWeights.GetColumn( 1 ) ),
					HiddenActivationPrime( HiddenLayer[2] ) * ( GlobalError * HiddenWeights.GetColumn( 2 ) )
				} );

				HiddenBiasErrors += HiddenError;
				InputWeightErrors += new Matrix<double>( new[] {
					Input * HiddenError[0],
					Input * HiddenError[1],
					Input * HiddenError[2]
				} );
			}

			OutputBiases += OutputBiasErrors.Apply( d => d / Inputs.Count() );
			HiddenBiases += HiddenBiasErrors.Apply( d => d / Inputs.Count() );
			HiddenWeights += HiddenWeightErrors.Apply( d => d / Inputs.Count() );
			InputWeights += InputWeightErrors.Apply( d => d / Inputs.Count() );
		}
*/