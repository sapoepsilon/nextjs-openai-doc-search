import type { NextRequest } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { codeBlock, oneLine } from 'common-tags';
import GPT3Tokenizer from 'gpt3-tokenizer';
import {
  Configuration,
  OpenAIApi,
  CreateModerationResponse,
  CreateEmbeddingResponse,
  ChatCompletionRequestMessage,
} from 'openai-edge';
import { OpenAIStream, StreamingTextResponse } from 'ai';
import { ApplicationError, UserError } from '@/lib/errors';

const openAiKey = process.env.OPENAI_KEY;
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

const config = new Configuration({
  apiKey: openAiKey,
});
const openai = new OpenAIApi(config);

export const runtime = 'edge';

export default async function handler(req: NextRequest) {
  try {
    if (!openAiKey) {
      throw new ApplicationError('Missing environment variable OPENAI_KEY');
    }

    if (!supabaseUrl) {
      throw new ApplicationError('Missing environment variable SUPABASE_URL');
    }

    if (!supabaseServiceKey) {
      throw new ApplicationError('Missing environment variable SUPABASE_SERVICE_ROLE_KEY');
    }

    console.log('Request received:', req.method, req.url);

    const requestData = await req.json();

    if (!requestData) {
      throw new UserError('Missing request data');
    }

    console.log('Request data:', requestData);

    const { prompt: query } = requestData;

    if (!query) {
      throw new UserError('Missing query in request data');
    }

    console.log('Query:', query);

    const supabaseClient = createClient(supabaseUrl, supabaseServiceKey);

    // Moderate the content to comply with OpenAI T&C
    const sanitizedQuery = query.trim();
    const moderationResponse: CreateModerationResponse = await openai
      .createModeration({ input: sanitizedQuery })
      .then((res) => res.json());

    console.log('Moderation response:', moderationResponse);

    if (!moderationResponse || !moderationResponse.results || !Array.isArray(moderationResponse.results)) {
      throw new ApplicationError('Invalid moderation response', moderationResponse);
    }

    const [results] = moderationResponse.results;

    if (results.flagged) {
      console.warn('Flagged content:', results.categories);
      throw new UserError('Flagged content', {
        flagged: true,
        categories: results.categories,
      });
    }

    // Create embedding from query
    const embeddingResponse = await openai.createEmbedding({
      model: 'text-embedding-ada-002',
      input: sanitizedQuery.replaceAll('\n', ' '),
    });

    console.log('Embedding response status:', embeddingResponse.status);

    if (embeddingResponse.status !== 200) {
      throw new ApplicationError('Failed to create embedding for question', embeddingResponse);
    }

    const embeddingData: CreateEmbeddingResponse = await embeddingResponse.json();

    console.log('Embedding data:', embeddingData);

    if (!embeddingData || !embeddingData.data || !Array.isArray(embeddingData.data) || embeddingData.data.length === 0) {
      throw new ApplicationError('Invalid embedding response', embeddingData);
    }

    const [{ embedding }] = embeddingData.data;

    const { error: matchError, data: pageSections } = await supabaseClient.rpc(
      'match_page_sections',
      {
        embedding,
        match_threshold: 0.78,
        match_count: 10,
        min_content_length: 50,
      }
    );

    console.log('Page sections:', pageSections);

    if (matchError) {
      throw new ApplicationError('Failed to match page sections', matchError);
    }

    const tokenizer = new GPT3Tokenizer({ type: 'gpt3' });
    let tokenCount = 0;
    let contextText = '';

    for (let i = 0; i < pageSections.length; i++) {
      const pageSection = pageSections[i];
      const content = pageSection.content;
      const encoded = tokenizer.encode(content);
      tokenCount += encoded.text.length;

      if (tokenCount >= 1500) {
        break;
      }

      contextText += `${content.trim()}\n---\n`;
    }

    console.log('Context text generated:', contextText);

    const prompt = codeBlock`
      ${oneLine`
        You are a very enthusiastic representative who loves
        to help people! Given the following sections from the Apple ShaderGrapher documentation, answer the question using only that information,
        outputted in markdown format.
      `}

      Context sections:
      ${contextText}

      Question: """
      ${sanitizedQuery}
      """

      Answer as markdown (including related code snippets if available):
    `

    const chatMessage: ChatCompletionRequestMessage = {
      role: 'user',
      content: prompt,
    }

    const response = await openai.createChatCompletion({
      model: 'gpt-4-turbo',
      messages: [chatMessage],
      max_tokens: 4096,
      temperature: 0.05,
      stream: true,
    });

    if (!response.ok) {
      const error = await response.json();
      console.error('Chat completion error:', error);
      throw new ApplicationError('Failed to generate completion', error);
    }

    // Transform the response into a readable stream
    const stream = OpenAIStream(response);

    console.log('Stream created successfully');

    // Return a StreamingTextResponse, which can be consumed by the client
    return new StreamingTextResponse(stream);
  } catch (err: unknown) {
    if (err instanceof UserError) {
      console.warn('User error:', err.message, err.data);
      return new Response(
        JSON.stringify({
          error: err.message,
          data: err.data,
        }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    } else if (err instanceof ApplicationError) {
      console.error('Application error:', err.message, err.data);
    } else {
      console.error('Unexpected error:', err);
    }

    // TODO: include more response info in debug environments
    return new Response(
      JSON.stringify({
        error: 'There was an error processing your request',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
