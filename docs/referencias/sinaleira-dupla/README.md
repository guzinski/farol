# Sinaleira dupla: seta de conversão ao lado do foco principal

Cruzamento gravado em 2026-11-19 (sessões `12-43-56` e `13-13-49`). A seta verde de conversão à
esquerda fica ao lado do foco principal, junto da placa amarela, e os dois estão à frente do carro.

- `seta-ao-lado-da-placa_*.jpg`: o braço com a placa, a seta e o foco principal.
- `mesmo-cruzamento-com-arvores_*.jpg`: o mesmo ponto na outra gravação, com árvores atrás.
- `seta-verde-quase-branca_*.png`: a caixa da seta ampliada. Acesa, ela sai quase branca, só com a borda verde-clara.

Antes, o app detectava a caixa da seta mas não lia cor nenhuma, e a contava como "em frente" por estar a
40% da largura da imagem. A leitura de verde pálido e a regra de luzes lado a lado tratam esses dois casos.
Os recortes `android-13-13-49-*` em `android/app/src/test/resources/parity` cobrem isso nos testes.
