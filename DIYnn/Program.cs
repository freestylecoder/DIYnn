using Freestylecoding.Math.Linear;

namespace DIYnn {
	internal class Program {
		/// <summary>Writes the Expected and Actual results of a Neural Network computation to the Console</summary>
		/// <param name="Expected">The expected result of a Neural Network computation</param>
		/// <param name="Actual">The expected result of a Neural Network computation</param>
		/// <remarks>
		///		If the values match, Actual is displayed in Green.
		///		Otherwise, Actual is displayed in Red.
		/// </remarks>
		private static void PrintColorful( bool Expected, bool Actual ) {
			Console.Write( $"Expected: {Expected}\tActual: " );
			Console.ForegroundColor =
				Expected == Actual
					? ConsoleColor.Green
					: ConsoleColor.Red;
			Console.WriteLine( Actual );
			Console.ResetColor();
		}

		static void Main() {
			// This is a trinary XOR where the 4th argument is ignored
			// THUS: Expected = Input[0] ^ Input[1] ^ Input[2]
			IEnumerable<(Vector<double> input, bool Expected)> inputs = new[] {
				(new Vector<double>( new double[] { 0.0, 0.0, 0.0, 0.0 } ), false),
				(new Vector<double>( new double[] { 0.0, 0.0, 0.0, 1.0 } ), false),
				(new Vector<double>( new double[] { 0.0, 0.0, 1.0, 0.0 } ), true ),
				(new Vector<double>( new double[] { 0.0, 0.0, 1.0, 1.0 } ), true ),
				(new Vector<double>( new double[] { 0.0, 1.0, 0.0, 0.0 } ), true ),
				(new Vector<double>( new double[] { 0.0, 1.0, 0.0, 1.0 } ), true ),
				(new Vector<double>( new double[] { 0.0, 1.0, 1.0, 0.0 } ), false),
				(new Vector<double>( new double[] { 0.0, 1.0, 1.0, 1.0 } ), false),
				(new Vector<double>( new double[] { 1.0, 0.0, 0.0, 0.0 } ), true ),
				(new Vector<double>( new double[] { 1.0, 0.0, 0.0, 1.0 } ), true ),
				(new Vector<double>( new double[] { 1.0, 0.0, 1.0, 0.0 } ), false),
				(new Vector<double>( new double[] { 1.0, 0.0, 1.0, 1.0 } ), false),
				(new Vector<double>( new double[] { 1.0, 1.0, 0.0, 0.0 } ), false),
				(new Vector<double>( new double[] { 1.0, 1.0, 0.0, 1.0 } ), false),
				(new Vector<double>( new double[] { 1.0, 1.0, 1.0, 0.0 } ), true ),
				(new Vector<double>( new double[] { 1.0, 1.0, 1.0, 1.0 } ), true ),
			};

			// Create the Neural Network
			NeuralNet<bool> NN = new NeuralNet<bool>( 4, 3, 2, o => o[0] > o[1] );
			NN.HiddenActivation = NeuralNet<double>.Sigmoid;
			NN.HiddenActivationPrime = NeuralNet<double>.SigmoidPrime;
			NN.OutputActivation = NeuralNet<double>.Sigmoid;
			NN.OutputActivationPrime = NeuralNet<double>.SigmoidPrime;

			// Test the initial state of the network and display the results
			int correct = 0;
			foreach( (Vector<double> input, bool Expected) in inputs ) {
				bool Actual = NN.Test( input );
				correct += Expected == Actual ? 1 : 0;
				PrintColorful( Expected, Actual );
			}

			Console.ReadKey();

			// Prepare the training algorithm
			int epoch = 0;
			Func<bool,Vector<double>> TransformExpected = Expected =>
				new Vector<double>(
					Expected ? new[] { 1.0, 0.0 } : new[] { 0.0, 1.0 }
				);

			// Keep training until it works...
			while( correct != inputs.Count() ) {
				// Every 1000 tries, update the status to the Console
				if( 0 == epoch % 1000 ) {
					Console.Clear();
					Console.Write( $"Epoch {epoch}\tCorrect: {correct}" );
					Console.WriteLine();

					foreach( (Vector<double> input, bool Expected) in inputs )
						PrintColorful( Expected, NN.Test( input ) );
					Console.WriteLine();

					NN.PrintNetwork();
				}

				// Train the network
				NN.Train( inputs, TransformExpected );

				// See how many are correct
				++epoch;
				correct = 0;
				foreach( (Vector<double> input, bool Expected) in inputs ) {
					correct += Expected == NN.Test( input ) ? 1 : 0;
				}
			}

			// Output the working results and final values of the NN
			Console.Clear();
			Console.WriteLine( $"{epoch} Epochs" );
			foreach( (Vector<double> input, bool Expected) in inputs )
				PrintColorful( Expected, NN.Test( input ) );

			Console.WriteLine();
			NN.PrintNetwork();

		}
	}
}