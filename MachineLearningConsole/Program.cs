using MachineLearning.ML;
using MachineLearning.Models;


// ExemploRegressao();

ExemploClassificacaoBinaria();

void ExemploClassificacaoBinaria()
{
    var trainer = new CreditoModelTrainer();

    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "aprovacao_credito.csv"));
    trainer.TreinarModelo();
    trainer.AvaliarModelo();
    trainer.AvaliarMelhorModelo();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinado_classificacao_binaria.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new CreditoModelPrecictor();
    predictor.CarregarModelo(pathModelo);

    var novoCredito = new CreditoInputData()
    {
        RendaMensal = 4200f,
        EstadoCivil = 1,
        NumeroDependentes = 2,
        PossuiVeiculo = 1,
        JaNegadoAntes = 0
    };

    var resultado = predictor.Prever(novoCredito);
    Console.WriteLine($"Aprovado? {(resultado.PredicaoAprovado ? "Sim" : "Não")}");
    Console.WriteLine($"Probabilidade: {resultado.Probability:P2}");
}

void ExemploRegressao()
{
    var trainer = new CasaModelTrainer();

    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "casas_treinamento.csv"));
    trainer.TreinarModelo();

    trainer.AvaliarModelo();
    trainer.AvaliarMelhorAlgoritino();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinado_regressao.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new CasaModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var casaNova = new CasaInputData()
    {
        Tamanho = 85f,
        Quartos = 3
    };

    var resultado = predictor.Prever(casaNova);
    Console.WriteLine("O valor da casa nova é: " + resultado.PrecoPrevisto);
}