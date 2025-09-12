using MachineLearning.ML;
using MachineLearning.Models;


// ExemploRegressao();
// ExemploClassificacaoBinaria();
ExemploClassificacaoMultiClasse();

void ExemploClassificacaoMultiClasse()
{
    var trainer = new PerfilAlunoModelTrainer();

    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "perfil_aluno_idiomas.csv"));
    trainer.TreinarModelo();

    trainer.AvaliarModelo();
    trainer.AvaliarMelhorModelo();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinado_classificacao_multiclasse.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new PerfilAlunoModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var novoAluno = new PerfilAlunoInputData()
    {
        NotaProficienciaGramatical = 6.5f,
        CompreensaoOral = 7.0f,
        NotaConversacao = 5.5f
    };

    var resultado = predictor.Prever(novoAluno);

    Console.WriteLine($"Perfil previsto: {resultado.PerfilPrevisto}");

    Console.WriteLine("Pontuação por perfil:");

    var perfis = new[] { "Iniciante", "Intermediário", "Avançado" };

    for (int cont = 0; cont < resultado.Score.Length; cont++)
    {
        Console.WriteLine($"{perfis[cont]}: {resultado.Score[cont]:P2}");
    }
}

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